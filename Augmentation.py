import os
import numpy as np
import albumentations as A
import cv2

patchescena = np.load("RESERVA/2025/11 07/CENA_PATCHES.npy")
patchesmascara = np.load("RESERVA/2025/11 07/MASCARA_PATCHES.npy")

pasta_saida = "RESERVA/2025/11 07/"

print(f"Shape dos patches CENA: {patchescena.shape}")
print(f"Shape dos patches MASCARA: {patchesmascara.shape}")

pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=0.4
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.05,
        contrast_limit=0.05,
        p=0.2
    ),

    A.OneOf([
        A.GaussNoise(
            std_range=(0.0, 0.01)),
        A.GaussianBlur(blur_limit=3),
    ], p=0.3),
])

cena_aumentados = []
mascara_aumentados = []

for cenas, mascaras in zip(patchescena,patchesmascara):

    aumento = pipeline(cenas=cenas, mascaras=mascaras)

    cena_aumentados.append(aumento["cenas"])
    mascara_aumentados.append(aumento["mascaras"])

cena_aumentados = np.array(cena_aumentados, dtype=np.float32)
mascara_aumentados = np.array(mascara_aumentados, dtype=np.float32)

final_cena = np.concatenate([patchescena, cena_aumentados], axis=0)
final_mascara = np.concatenate([patchesmascara, mascara_aumentados], axis=0)

print(f"shape do aumento_cena {cena_aumentados.shape}")
print(f"shape do aumento_mascara {mascara_aumentados.shape}")

np.save(os.path.join(pasta_saida, "CENA_PATCH_AUG.npy"), cena_aumentados)
np.save(os.path.join(pasta_saida, "MASCARA_PATCH_AUG.npy"), mascara_aumentados)
print("\n")
print(f"shape do aumento_cena {final_cena.shape}")
print(f"shape do aumento_mascara {final_mascara.shape}")


print("Augmentation feita.")