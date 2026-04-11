import os
import gc
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from sklearn.model_selection import GroupKFold
from google.colab import drive



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth ativado.")
    except RuntimeError:
        print(" GPU já inicializada. Reinicie o ambiente se quiser alterar memory_growth.")


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



# INICIALIZAÇÃO DOS DADOS
THRESHOLD = 0.5
TRUE_THRESHOLD = 0.5
EPS = 1e-7

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

caminho_raiz = "/content/drive/MyDrive/DATASET_OTIMIZADO"
caminho_resultados = "/content/drive/MyDrive/ResultadoTHR0.5_com30epocas"

if not os.path.exists(caminho_raiz):
    raise ValueError("Pasta raiz do dataset não encontrada!")

os.makedirs(caminho_resultados, exist_ok=True)


X_list, y_list, groups = [], [], []
meta_list = []
scene_id = 0
input_shape = None

print("Carregando dataset")

for ano in sorted(os.listdir(caminho_raiz)):
    path_ano = os.path.join(caminho_raiz, ano)
    if not os.path.isdir(path_ano) or "Resultados" in ano:
        continue

    for cena in sorted(os.listdir(path_ano)):
        path_cena = os.path.join(path_ano, cena)

        x_path = os.path.join(path_cena, "CENA_PATCH_AUG.npy")
        y_path = os.path.join(path_cena, "MASCARA_PATCH_AUG.npy")

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            continue

        x_data = np.load(x_path).astype(np.float32)
        y_data = np.load(y_path).astype(np.float32)

        if input_shape is None:
            input_shape = x_data.shape[1:]

        if x_data.max() > 1.0:
            x_data /= 255.0

        X_list.append(x_data)
        y_list.append(y_data)

        groups.extend([scene_id] * x_data.shape[0])

        nome_cena = f"{ano}/{cena}"
        meta_list.extend([nome_cena] * x_data.shape[0])

        scene_id += 1

if len(X_list) == 0:
    raise ValueError("ERRO NENHUM .NPY ENCONTRADO")

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
groups = np.array(groups)
meta_data = np.array(meta_list)

print(f"Total de patches: {X.shape[0]}")
print(f"Total de cenas: {scene_id}")
print(f"Shape entrada: {input_shape}")

del X_list, y_list, meta_list
gc.collect()



# OTIMIZAÇÃO DA RAM E METRICAS DE AVALIAÇÃO
class RamDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, idx, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.idx = np.array(idx)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, i):
        batch = self.idx[i*self.batch_size:(i+1)*self.batch_size]
        return self.x[batch], self.y[batch]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

def _to_binary(y_true, y_pred, thr_pred=THRESHOLD, thr_true=TRUE_THRESHOLD):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_bin = tf.cast(y_true > thr_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > thr_pred, tf.float32)
    return y_true_bin, y_pred_bin

class BinaryConfusionMetric(tf.keras.metrics.Metric):
    def __init__(self, name="bin_metric", threshold=THRESHOLD, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin, y_pred_bin = _to_binary(y_true, y_pred, thr_pred=self.threshold, thr_true=TRUE_THRESHOLD)

        tp = tf.reduce_sum(y_true_bin * y_pred_bin)
        fp = tf.reduce_sum((1.0 - y_true_bin) * y_pred_bin)
        fn = tf.reduce_sum(y_true_bin * (1.0 - y_pred_bin))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

class BinaryPrecision(BinaryConfusionMetric):
    def __init__(self, threshold=THRESHOLD, name="precision_bin", **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)

    def result(self):
        return self.tp / (self.tp + self.fp + EPS)

class BinaryRecall(BinaryConfusionMetric):
    def __init__(self, threshold=THRESHOLD, name="recall_bin", **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)

    def result(self):
        return self.tp / (self.tp + self.fn + EPS)

class BinaryF1(BinaryConfusionMetric):
    def __init__(self, threshold=THRESHOLD, name="f1_bin", **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)

    def result(self):
        prec = self.tp / (self.tp + self.fp + EPS)
        rec  = self.tp / (self.tp + self.fn + EPS)
        return (2.0 * prec * rec) / (prec + rec + EPS)

class BinaryDice(BinaryConfusionMetric):
    def __init__(self, threshold=THRESHOLD, name="dice_bin", **kwargs):
        super().__init__(name=name, threshold=threshold, **kwargs)

    def result(self):
        # Dice = 2TP / (2TP + FP + FN)
        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + EPS)

def iou_metric_bin(y_true, y_pred):
    y_true_bin, y_pred_bin = _to_binary(y_true, y_pred, thr_pred=THRESHOLD, thr_true=TRUE_THRESHOLD)
    inter = tf.reduce_sum(y_true_bin * y_pred_bin)
    union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - inter
    return (inter + EPS) / (union + EPS)

def tversky(y_true, y_pred, alpha=0.3, beta=0.7):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

def focal_tversky_loss(y_true, y_pred):
    return tf.pow((1 - tversky(y_true, y_pred)), 0.75)



## MODELO U-NET
def get_unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    def bloco(x, f):
        x = tf.keras.layers.Conv2D(f, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(f, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    c1 = bloco(inputs, 16); p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = bloco(p1, 32); p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = bloco(p2, 64); p3 = tf.keras.layers.MaxPooling2D()(c3)
    c4 = bloco(p3, 128); p4 = tf.keras.layers.MaxPooling2D()(c4)

    b = bloco(p4, 256)
    b = tf.keras.layers.Dropout(0.3)(b)

    u4 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(b)
    u4 = tf.keras.layers.concatenate([u4, c4])
    c5 = bloco(u4, 128)

    u3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c5)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c6 = bloco(u3, 64)

    u2 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c6)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c7 = bloco(u2, 32)

    u1 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c7)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c8 = bloco(u1, 16)

    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(c8)
    return tf.keras.Model(inputs, out)

class CenaMonitor(Callback):
    def __init__(self, val_scenes):
        super().__init__()
        self.val_scenes = val_scenes

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0 or epoch % 10 == 0:
            print(f"\n[Época {epoch+1}] Validação nestas cenas:")
            print(f"{self.val_scenes}")




# TREINAMENTO
gkf = GroupKFold(n_splits=5)
fold = 1
results = []

for train_idx, val_idx in gkf.split(X, y, groups):
    print(f"\n{'='*40}")
    print(f"===== INICIANDO FOLD {fold} =====")
    print(f"{'='*40}")

    cenas_treino = np.unique(meta_data[train_idx])
    cenas_validacao = np.unique(meta_data[val_idx])

    print(f"\nTOTAL DE CENAS DE TREINO: {len(cenas_treino)}")
    print(f"\nTOTAL DE CENAS DE VALIDAÇÃO: {len(cenas_validacao)}")
    print(f"Lista (alvo do teste): {cenas_validacao}\n")

    train_gen = RamDataGenerator(X, y, train_idx, batch_size=32, shuffle=True)
    val_gen   = RamDataGenerator(X, y, val_idx, batch_size=32, shuffle=False)

    model = get_unet(input_shape)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=50 * len(train_gen),
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    dice_bin = BinaryDice(threshold=THRESHOLD, name="dice_bin")
    prec_bin = BinaryPrecision(threshold=THRESHOLD, name="precision_bin")
    rec_bin  = BinaryRecall(threshold=THRESHOLD, name="recall_bin")
    f1_bin   = BinaryF1(threshold=THRESHOLD, name="f1_bin")

    model.compile(
        optimizer=optimizer,
        loss=focal_tversky_loss,
        metrics=[dice_bin, prec_bin, rec_bin, f1_bin, iou_metric_bin],
        jit_compile=True
    )

    ckpt_path = os.path.join(caminho_resultados, f"modelo_fold_{fold}.weights.h5")
    log_path  = os.path.join(caminho_resultados, f"log_fold_{fold}.csv")

    callbacks = [
        ModelCheckpoint(filepath=ckpt_path, monitor='val_dice_bin',
                        save_best_only=True, save_weights_only=True, mode='max', verbose=0),
        EarlyStopping(monitor='val_dice_bin', patience=12, mode='max', restore_best_weights=True),
        CSVLogger(log_path),
        CenaMonitor(cenas_validacao)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        verbose=1,
        callbacks=callbacks
    )


    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    scores = model.evaluate(val_gen, verbose=0)
    fold_row = {
        "fold": fold,
        "val_scenes": len(cenas_validacao),
        "loss": float(scores[0]),
        "dice_bin": float(scores[1]),
        "precision": float(scores[2]),
        "recall": float(scores[3]),
        "f1": float(scores[4]),
        "iou": float(scores[5]),
    }
    results.append(fold_row)

    print(
        f"Fold {fold} -> "
        f"Dice(bin): {fold_row['dice_bin']:.4f} | "
        f"P: {fold_row['precision']:.4f} | "
        f"R: {fold_row['recall']:.4f} | "
        f"F1: {fold_row['f1']:.4f} | "
        f"IoU: {fold_row['iou']:.4f}"
    )

    del model, train_gen, val_gen
    tf.keras.backend.clear_session()
    gc.collect()
    fold += 1



# RELATÓRIO FINAL
print("\n" + "="*60)
print("RELATÓRIO FINAL (por fold)")
print("="*60)

for r in results:
    print(
        f"Fold {r['fold']} | cenas_val={r['val_scenes']} | "
        f"Dice={r['dice_bin']:.4f} | P={r['precision']:.4f} | "
        f"R={r['recall']:.4f} | F1={r['f1']:.4f} | IoU={r['iou']:.4f}"
    )

def _mean_std(key):
    vals = np.array([r[key] for r in results], dtype=np.float32)
    return float(vals.mean()), float(vals.std(ddof=0))

dice_m, dice_s = _mean_std("dice_bin")
p_m, p_s       = _mean_std("precision")
r_m, r_s       = _mean_std("recall")
f1_m, f1_s     = _mean_std("f1")
iou_m, iou_s   = _mean_std("iou")

print("\n" + "="*60)
print(f"THRESHOLD (y_pred): {THRESHOLD}")
print("MÉDIA ± DP (5 folds)")
print("="*60)
print(f"Dice(bin): {dice_m:.4f} ± {dice_s:.4f}")
print(f"Precision: {p_m:.4f} ± {p_s:.4f}")
print(f"Recall:    {r_m:.4f} ± {r_s:.4f}")
print(f"F1-score:  {f1_m:.4f} ± {f1_s:.4f}")
print(f"IoU:       {iou_m:.4f} ± {iou_s:.4f}")