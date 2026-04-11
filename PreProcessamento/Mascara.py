import os
import numpy as np
import rasterio

RAIZ = "BANDAS_BRUTAS"
PASTA_SAIDA = "MASCARAS_AGUA"

PERC_B6 = 8
NDWI_MIN = -0.12

BAND_GREEN = "B3"
BAND_NIR = "B5"
BAND_B6 = "B6"

os.makedirs(PASTA_SAIDA, exist_ok=True)

def ler_banda(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = 0
    return arr, profile


def calcular_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-6)

for ano in sorted(os.listdir(RAIZ)):
    pasta_ano = os.path.join(RAIZ, ano)
    if not os.path.isdir(pasta_ano):
        continue

    for dia in sorted(os.listdir(pasta_ano)):
        pasta_dia = os.path.join(pasta_ano, dia)
        if not os.path.isdir(pasta_dia):
            continue

        # localizar bandas
        arquivos = os.listdir(pasta_dia)

        def achar_banda(nome):
            for f in arquivos:
                if nome in f.upper() and f.lower().endswith(".tif"):
                    return os.path.join(pasta_dia, f)
            return None

        path_green = achar_banda(BAND_GREEN)
        path_nir   = achar_banda(BAND_NIR)
        path_b6    = achar_banda(BAND_B6)

        if not all([path_green, path_nir, path_b6]):
            print(f"[AVISO] Bandas faltando em {ano}/{dia}")
            continue

        print(f"[INFO] Processando {ano}/{dia}")

        green, profile = ler_banda(path_green)
        nir, _ = ler_banda(path_nir)
        b6, _ = ler_banda(path_b6)

        ndwi = calcular_ndwi(green, nir)

        valid_b6 = b6 > 0

        if np.sum(valid_b6) < 100:
            print("Poucos pixels válidos")
            continue

        thr_b6 = np.percentile(b6[valid_b6], PERC_B6)

        mask_b6 = b6 <= thr_b6
        mask_sombra = ndwi < NDWI_MIN

        mask_agua = mask_b6 & (~mask_sombra)

        mask_agua = mask_agua.astype(np.uint8)

        print(f"  B6 thr={thr_b6:.2f} | NDWI_MIN={NDWI_MIN}")

        profile_out = profile.copy()
        profile_out.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,
            compress="lzw"
        )

        nome_saida = f"MASCARA_AGUA_{ano}_{dia.replace(' ', '_')}.tif"
        path_saida = os.path.join(PASTA_SAIDA, nome_saida)

        with rasterio.open(path_saida, "w", **profile_out) as dst:
            dst.write(mask_agua, 1)

print("Processamento concluído.")
