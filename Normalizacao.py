import rasterio
import json
import numpy as np

def normalizar_com_estatisticas(img, stats):
    img = img.astype(np.float32)

    p_low = stats["low"]
    p_high = stats["high"]

    den = p_high - p_low
    if den < 1e-6:
        return np.zeros_like(img, dtype=np.float32)

    mask = ~np.isnan(img)

    img_norm = np.zeros_like(img, dtype=np.float32)
    img_norm[mask] = (img[mask] - p_low) / den
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm

input_tif = "CAMINHO/CENA"
output_tif = "CAMINHO/CENA NORMALIZADA"
stats_path = "stats_normalizacao_global_por_banda.json" #JSON global usado para todas as cenas

with open(stats_path) as f:
    stats = json.load(f)

with rasterio.open(input_tif) as src:
    profile = src.profile.copy()
    img = src.read()

    img_norm = normalizar_com_estatisticas(img, stats)

    profile.update(
        dtype=rasterio.float32,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(img_norm)

print("Normalização aplicada com estatísticas globais fixas!")
