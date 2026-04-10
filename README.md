# Estimativa de Volume Hídrico com Deep Learning (U-Net)

Este repositório apresenta um projeto de **estimativa de volume hídrico em reservatórios** utilizando técnicas de **Deep Learning**, com foco na arquitetura **U-Net** aplicada a imagens de sensoriamento remoto.

## Sobre o Projeto

O objetivo deste trabalho é desenvolver um modelo capaz de **estimar automaticamente o volume de água de um reservatório**, reduzindo a dependência de métodos manuais tradicionalmente utilizados por órgãos de monitoramento hídrico.

Como estudo de caso, foi utilizado o **Açude de Orós**, localizado no estado do Ceará, um dos principais reservatórios da região Nordeste.

Os resultados obtidos nesse projeto foram transformados em um artigo acadêmico, submetido ao VII Forum Internacional do Semiárido.

## Link para o artigo completo

Link: "placeholder"

### 🔹 Etapas principais:
- Coleta de imagens (2021–2025)
- Pré-processamento dos dados
- Treinamento da rede neural
- Segmentação do espelho d'água
- Estimativa do volume hídrico com base na área identificada

### Métricas de segmentação:
- **Coeficiente de Dice:** 93,4%
- **Intersection over Union (IoU):** 86,6%

### Métricas de estimativa de volume:
- **MAPE:** 5,54%
- **RMSE:** 6,99%
- **Bias:** -0,14%
- **NSE:** 0,950

## Aplicações

- Monitoramento hídrico automatizado
- Apoio à gestão de recursos hídricos
- Redução de custos operacionais
- Escalabilidade para outros reservatórios
