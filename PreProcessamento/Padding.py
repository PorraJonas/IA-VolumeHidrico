import numpy as np
import rasterio as rs
import math
import os


img_path = "CAMINHO/CENA"
mask_path = "CAMINHO/MASCARA"
saida_img = "CAMINHO_SAIDA/CENA"
saida_mask = "CAMINHO_SAIDA/MASCARA"

with rs.open(img_path) as src:
    img = src.read()
    meta = src.meta.copy()

with rs.open(mask_path) as src:
    nuvem = src.read(1)

channels, height, width = img.shape
print(f"Shape Original: {height} x {width}")

# cálculo do padding
target_h = math.ceil(height / 128) * 128
target_w = math.ceil(width / 128) * 128
pad_h = target_h - height
pad_w = target_w - width


img_padded = np.pad(img,((0, 0), (0, pad_h), (0, pad_w)),mode="constant",constant_values=np.nan)

nuvem_padded = np.pad(nuvem,((0, pad_h), (0, pad_w)),mode="constant",constant_values=1)

print(f"Shape Padded: {target_h} x {target_w}")

# atualiza metadata
meta.update({
    "height": target_h,
    "width": target_w,
    "nodata": np.nan,
    "dtype": "float32"
})


with rs.open(saida_img, "w", **meta) as dst:
    dst.write(img_padded)


meta_mask = meta.copy()
meta_mask.update({
    "height": target_h,
    "width": target_w,
    "nodata": 1
})

with rs.open(saida_mask, "w", **meta_mask) as dst:
    dst.write(nuvem_padded, 1)

print("Padding finalizado.")
