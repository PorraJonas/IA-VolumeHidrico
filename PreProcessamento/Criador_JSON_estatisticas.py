import numpy as np
import rasterio
import json
from collections import defaultdict

def calcular_percentis_globais_por_banda(lista_tifs, pmin=2,pmax=98):
    acumulado_por_banda = defaultdict(list)

    for tif in lista_tifs:

        with rasterio.open(tif) as src:
            nodata = src.nodata
            n_bandas = src.count

            for b in range(1, n_bandas + 1):
                banda = src.read(b).astype(np.float32)

                if nodata is not None:
                    mask = banda != nodata
                else:
                    mask = ~np.isnan(banda)

                dados = banda[mask]

                if dados.size == 0:
                    continue

                acumulado_por_banda[b].append(dados)

    estatisticas = {}

    for b, listas in acumulado_por_banda.items():
        todos = np.concatenate(listas)

        estatisticas[f"B{b}"] = {
            "pmin": pmin,
            "pmax": pmax,
            "low": float(np.percentile(todos, pmin)),
            "high": float(np.percentile(todos, pmax))
        }

        print(
            f"Banda {b}: "
            f"low={estatisticas[f'B{b}']['low']:.6f}, "
            f"high={estatisticas[f'B{b}']['high']:.6f}"
        )

    return estatisticas

lista_cenas = [
    "Colocar o caminho individual para todas as cenas do dataset"
]

stats = calcular_percentis_globais_por_banda(
    lista_cenas,
    pmin=2,
    pmax=98
)

with open("stats_normalizacao_global_por_banda.json", "w") as f:
    json.dump(stats, f, indent=4)

print("Arquivo stats_normalizacao_global_por_banda.json salvo.")