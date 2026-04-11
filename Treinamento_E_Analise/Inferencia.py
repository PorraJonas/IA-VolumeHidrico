import os
import numpy as np
import pandas as pd
import tensorflow as tf
import rasterio
from scipy.ndimage import binary_dilation

CAMINHO_RAIZ = "/content/drive/MyDrive/MASCARA E CENA"
CAMINHO_RESULTADOS = "/content/drive/MyDrive/Resultados_Treino_definitivo"

ARQ_CENA_TIF = "CENA.tif"
ARQ_MASK_TIF = "MASCARA.tif"

FOLD = 1
MODELO_PATH = os.path.join(CAMINHO_RESULTADOS, f"modelo_fold_{FOLD}.keras")

BATCH = 8
# Normalização da entrada do modelo (deve bater o máximo possível com o treino)
P_LOW_NORM, P_HIGH_NORM = 2, 98 

THR_GRID = np.linspace(0.01, 0.99, 99)   # procura THR que melhor ACERTA ÁREA
DESEMPATE_POR_DICE = True                # se empatar no erro de área, escolhe maior Dice

USAR_DILATACAO = True
DILAT_ITER = 1   # 0=desliga, 1 ou 2 normalmente bastam

OUT_CSV = os.path.join(CAMINHO_RESULTADOS, f"pixels_agua_fold_{FOLD}_thr_area.csv")

def ler_tif_hwc(path_tif: str) -> np.ndarray:
    """Lê GeoTIFF e devolve (H,W,C) float32."""
    with rasterio.open(path_tif) as src:
        arr = src.read()  # (C,H,W)
    return np.transpose(arr, (1, 2, 0)).astype(np.float32)

def preprocess_mask(mask_raw: np.ndarray) -> np.ndarray:

    if mask_raw.ndim == 3:
        mask = mask_raw[..., 0]
    else:
        mask = mask_raw

    mx = np.nanmax(mask)
    if mx > 1.5:      # ex: 0/255
        mask = (mask > (mx * 0.5)).astype(np.float32)
    else:             # ex: 0/1
        mask = (mask > 0.5).astype(np.float32)
    return mask

def preprocess_para_modelo(img_raw: np.ndarray, canais_modelo: int, p_low=2, p_high=98) -> np.ndarray:

    img = img_raw[..., :canais_modelo].astype(np.float32)
    img = np.where(img < 0, 0, img)

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[-1]):
        lo = np.nanpercentile(img[..., c], p_low)
        hi = np.nanpercentile(img[..., c], p_high)
        if hi > lo:
            out[..., c] = (img[..., c] - lo) / (hi - lo)
        else:
            out[..., c] = 0.0
    return np.clip(out, 0, 1)

def inferencia_sliding_window_hann(img: np.ndarray, model, patch: int, stride: int, batch_size=8, verbose=0) -> np.ndarray:
    H, W, C = img.shape

    def padded_size(L):
        if L <= patch: return patch
        rem = (L - patch) % stride
        return L if rem == 0 else (L + (stride - rem))

    H_pad = padded_size(H)
    W_pad = padded_size(W)
    pad_h = H_pad - H
    pad_w = W_pad - W

    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    prob  = np.zeros((H_pad, W_pad), dtype=np.float32)
    count = np.zeros((H_pad, W_pad), dtype=np.float32)

    w1 = np.hanning(patch).astype(np.float32)
    w2 = np.outer(w1, w1).astype(np.float32)
    w2 = w2 / (w2.max() + 1e-8)

    patches, coords = [], []

    def flush():
        nonlocal patches, coords
        if not patches:
            return
        batch = np.stack(patches, axis=0)
        pred = model.predict(batch, verbose=verbose)
        if pred.ndim == 4:
            pred = pred[..., 0]
        for (y, x), pm in zip(coords, pred):
            pm = pm.astype(np.float32)
            prob[y:y+patch, x:x+patch]  += pm * w2
            count[y:y+patch, x:x+patch] += w2
        patches, coords = [], []

    for y in range(0, H_pad - patch + 1, stride):
        for x in range(0, W_pad - patch + 1, stride):
            patches.append(img_pad[y:y+patch, x:x+patch, :])
            coords.append((y, x))
            if len(patches) >= batch_size:
                flush()
    flush()

    prob /= (count + 1e-8)
    return prob[:H, :W]

def sigmoid_if_needed(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if (mn < 0.0) or (mx > 1.0):
        x = 1.0 / (1.0 + np.exp(-x))
    return x

def dice_bool(pred: np.ndarray, gt: np.ndarray, eps=1e-7) -> float:
    inter = int((pred & gt).sum())
    return float((2*inter + eps) / (pred.sum() + gt.sum() + eps))

def escolher_thr_area(prob_map: np.ndarray, gt: np.ndarray):
    """
    Escolhe THR que minimiza |agua_pred - agua_gt|.
    Desempata pelo maior Dice (opcional).
    """
    agua_gt = int(gt.sum())
    best = None

    for t in THR_GRID:
        pred = (prob_map >= t)
        if USAR_DILATACAO and DILAT_ITER > 0:
            pred = binary_dilation(pred, iterations=DILAT_ITER)

        agua_pred = int(pred.sum())
        erro_abs = abs(agua_pred - agua_gt)

        d = dice_bool(pred, gt)

        if best is None:
            best = (float(t), int(erro_abs), float(d), int(agua_pred), int(agua_gt))
        else:
            bt, berr, bd, bpred, bgt = best
            if erro_abs < berr:
                best = (float(t), int(erro_abs), float(d), int(agua_pred), int(agua_gt))
            elif erro_abs == berr and DESEMPATE_POR_DICE and d > bd:
                best = (float(t), int(erro_abs), float(d), int(agua_pred), int(agua_gt))

    return best  # (thr, erro_abs, dice, agua_pred, agua_gt)




# EXECUÇÃO
if not os.path.exists(CAMINHO_RAIZ):
    raise FileNotFoundError(f"Não achei: {CAMINHO_RAIZ}")
if not os.path.exists(MODELO_PATH):
    raise FileNotFoundError(f"Não achei: {MODELO_PATH}")

print("Carregando modelo:", MODELO_PATH)
model = tf.keras.models.load_model(MODELO_PATH, compile=False)

PATCH = model.input_shape[1]
STRIDE = PATCH // 2
CANAIS_MODELO = model.input_shape[-1]

print("PATCH:", PATCH, "| STRIDE:", STRIDE, "| CANAIS_MODELO:", CANAIS_MODELO)

rows = []

for ano in sorted(os.listdir(CAMINHO_RAIZ)):
    path_ano = os.path.join(CAMINHO_RAIZ, ano)
    if not os.path.isdir(path_ano):
        continue

    for cena in sorted(os.listdir(path_ano)):
        path_cena = os.path.join(path_ano, cena)
        if not os.path.isdir(path_cena):
            continue

        x_tif = os.path.join(path_cena, ARQ_CENA_TIF)
        y_tif = os.path.join(path_cena, ARQ_MASK_TIF)

        if not (os.path.exists(x_tif) and os.path.exists(y_tif)):
            continue

        cena_id = f"{ano}/{cena}"

        img_raw = ler_tif_hwc(x_tif)
        mask_raw = ler_tif_hwc(y_tif)
        gt = (preprocess_mask(mask_raw) > 0.5)

        img_model = preprocess_para_modelo(
            img_raw, CANAIS_MODELO,
            p_low=P_LOW_NORM, p_high=P_HIGH_NORM
        )

        prob_map = inferencia_sliding_window_hann(
            img_model, model, patch=PATCH, stride=STRIDE,
            batch_size=BATCH, verbose=0
        )
        prob_map = sigmoid_if_needed(prob_map)

        thr, erro_abs, d_at_thr, agua_pred_best, agua_gt = escolher_thr_area(prob_map, gt)

        pred = (prob_map >= thr)
        if USAR_DILATACAO and DILAT_ITER > 0:
            pred = binary_dilation(pred, iterations=DILAT_ITER)

        tp = int((pred & gt).sum())
        fp = int((pred & (~gt)).sum())
        fn = int(((~pred) & gt).sum())

        agua_pred = int(pred.sum())
        diff = agua_pred - agua_gt
        erro_pct = (diff / (agua_gt + 1e-8)) * 100.0

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        dice_val  = dice_bool(pred, gt)

        rows.append({
            "cena": cena_id,
            "fold": FOLD,
            "thr_area": thr,
            "agua_gt_pixels": int(agua_gt),
            "agua_pred_pixels": int(agua_pred),
            "diff_pixels": int(diff),
            "erro_pct_area": float(erro_pct),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "dice": float(dice_val),
            "dilatacao": int(DILAT_ITER if USAR_DILATACAO else 0),
            "p_low_norm": P_LOW_NORM,
            "p_high_norm": P_HIGH_NORM,
        })

        # print compacto
        print(f"{cena_id} | THR={thr:.2f} | GT={agua_gt} | IA={agua_pred} | diff={diff} | Dice={dice_val:.4f} | Prec={precision:.3f} | Rec={recall:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print("\nCSV salvo em:", OUT_CSV)
