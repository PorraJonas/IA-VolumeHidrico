import os
import numpy as np
import rasterio


def calcular_indice(banda_a, banda_b, nodata=0):
    banda_a = banda_a.astype(np.float32)
    banda_b = banda_b.astype(np.float32)

    indice = np.full_like(banda_a, nodata, dtype=np.float32)

    mask = (
        (banda_a != nodata) &
        (banda_b != nodata) &
        ((banda_a + banda_b) != 0)
    )

    indice[mask] = (
        (banda_a[mask] - banda_b[mask]) /
        (banda_a[mask] + banda_b[mask])
    )

    return indice

def gerar_ndvi(nir, red, nodata=0):
    return calcular_indice(nir, red, nodata)


def gerar_ndwi(green, nir, nodata=0):
    return calcular_indice(green, nir, nodata)

def gerar_imagem_7bandas(
    caminho_blue,
    caminho_green,
    caminho_red,
    caminho_nir,
    caminho_swir,
    saida_tif,
    nodata=0
):
    with rasterio.open(caminho_blue) as src:
        blue = src.read(1)
        perfil = src.profile.copy()

    green = rasterio.open(caminho_green).read(1)
    red = rasterio.open(caminho_red).read(1)
    nir = rasterio.open(caminho_nir).read(1)
    swir = rasterio.open(caminho_swir).read(1)

    # Padroniza NoData
    bandas = [blue, green, red, nir, swir]
    bandas = [np.nan_to_num(b.astype(np.float32), nan=nodata) for b in bandas]
    blue, green, red, nir, swir = bandas

    # Índices
    ndvi = gerar_ndvi(nir, red, nodata)
    ndwi = gerar_ndwi(green, nir, nodata)

    # Stack final
    stack = np.stack(
        [blue, green, red, nir, swir, ndvi, ndwi],
        axis=0
    )

    perfil.update(
        count=7,
        dtype=np.float32,
        nodata=nodata,
        compress="deflate",
        predictor=2
    )

    with rasterio.open(saida_tif, "w", **perfil) as dst:
        dst.write(stack)

    print(f"Gerado: {saida_tif}")

def processar_pasta_cenas(pasta_entrada, pasta_saida, nodata=0):
    os.makedirs(pasta_saida, exist_ok=True)

    for cena in sorted(os.listdir(pasta_entrada)):
        caminho_cena = os.path.join(pasta_entrada, cena)
        if not os.path.isdir(caminho_cena):
            continue

        print(f"Processando {cena}")

        arquivos = os.listdir(caminho_cena)

        def buscar(sufixo):
            return os.path.join(
                caminho_cena,
                next(a for a in arquivos if a.endswith(sufixo))
            )

        try:
            blue = buscar("_SR_B2.TIF")
            green = buscar("_SR_B3.TIF")
            red = buscar("_SR_B4.TIF")
            nir = buscar("_SR_B5.TIF")
            swir = buscar("_SR_B6.TIF")
        except StopIteration:
            print(f"Bandas faltando em {cena}")
            continue

        saida = os.path.join(
            pasta_saida,
            f"{cena}_FULL.tif"
        )

        gerar_imagem_7bandas(
            blue, green, red, nir, swir,
            saida_tif=saida,
            nodata=nodata
        )

if __name__ == "__main__":
    processar_pasta_cenas(
        pasta_entrada="ANO_XXXX",
        pasta_saida="IMAGENS_XXXX",
        nodata=0
    )
