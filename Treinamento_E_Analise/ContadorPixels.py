import rasterio
import numpy as np

with rasterio.open('MASCARA.tif') as src:
    # Lê a primeira banda da imagem
    data = src.read(1)

    # Se o 'NoData' estiver definido no arquivo, use-o:
    if src.nodata is not None:
        valid_pixels = np.count_nonzero(data != src.nodata)
    else:
        # Caso contrário, conta todos os pixels que não são pretos (valor 0)
        valid_pixels = np.count_nonzero(data > 0)

print(f"Total de pixels válidos: {valid_pixels}")