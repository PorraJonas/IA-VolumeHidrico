import numpy as np
import rasterio as rs
import os

img_path = "CAMINHO/CENA"
mask_path = "CAMINHO/MASCARA"

pasta_saida = "PASTA DE SAIDA"
os.makedirs(pasta_saida, exist_ok=True)

patch_size = 128
stride = 96
nodata_value = 0
max_nodata_frac = 0.3


def patch_valido(patch_cena, patch_mask, max_nodata_frac=0.3, max_cloud_frac=0.3):

    # pixels inválidos se QUALQUER banda for NaN
    nodata_mask = np.isnan(patch_cena).any(axis=-1)
    frac_nodata = nodata_mask.mean()

    # nuvem explícita
    cloud_mask = patch_mask == 1
    frac_cloud = cloud_mask.mean()

    return (frac_nodata <= max_nodata_frac) and (frac_cloud <= max_cloud_frac)

with rs.open(img_path) as src:
    img = src.read() 

img = np.moveaxis(img, 0, -1).astype(np.float32) 
H, W, C = img.shape


with rs.open(mask_path) as src:
    mask = src.read(1) 

mask = mask.astype(np.float32)

assert img.shape[:2] == mask.shape, "Cena e máscara têm dimensões diferentes!"


patches_cena = []
patches_mascara = []
descartados = 0
patches_totais = 0

for i in range(0, H - patch_size + 1, stride):
    for j in range(0, W - patch_size + 1, stride):
        patch_cena = img[i:i+patch_size, j:j+patch_size, :]
        patch_mask = mask[i:i+patch_size, j:j+patch_size]

        patches_totais += 1

        if patch_valido(
            patch_cena,
            patch_mask,
            max_nodata_frac=0.3,
            max_cloud_frac=0.3
        ):
            patches_cena.append(patch_cena)
            patches_mascara.append(patch_mask[..., np.newaxis])
        else:
            descartados += 1

patches_cena = np.array(patches_cena, dtype=np.float32)
patches_mascara = np.array(patches_mascara, dtype=np.float32)

print(f"Patches totais: {patches_totais}")
print(f"Patches válidos: {len(patches_cena)}")
print(f"Patches descartados: {descartados}")
print(f"Shape CENA: {patches_cena.shape}")
print(f"Shape MÁSCARA: {patches_mascara.shape}")

np.save(os.path.join(pasta_saida, "CENA_PATCHES.npy"), patches_cena)
np.save(os.path.join(pasta_saida, "MASCARA_PATCHES.npy"), patches_mascara)

print("Patches de cena e máscara salvos.")
