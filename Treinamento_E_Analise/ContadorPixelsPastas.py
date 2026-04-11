import os
import rasterio
import numpy as np
import pandas as pd

pasta_entrada = r"PASTA GERAL"
arquivo_saida = "contagem_pixels.csv"

resultados = []

# Percorre todos os arquivos da pasta
for arquivo in os.listdir(pasta_entrada):
    if arquivo.lower().endswith(".tif"):
        caminho_arquivo = os.path.join(pasta_entrada, arquivo)

        with rasterio.open(caminho_arquivo) as src:
            data = src.read(1)

            if src.nodata is not None:
                valid_pixels = np.count_nonzero(data != src.nodata)
            else:
                valid_pixels = np.count_nonzero(data > 0)

        resultados.append([arquivo, valid_pixels])
        print(f"{arquivo} -> {valid_pixels} pixels")


df = pd.DataFrame(resultados, columns=["Nome do arquivo", "qnt pixels"])
df.to_csv(arquivo_saida, sep=";", index=False)

print("\nArquivo CSV gerado com sucesso!")
