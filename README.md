# Estimativa de Volume Hídrico com Deep Learning (U-Net)

## Este repositório apresenta um projeto de estimativa de volume hídrico em reservatórios utilizando técnicas de Deep Learning, com foco na arquitetura U-Net aplicada a imagens de sensoriamento remoto.

Sobre o Projeto

O objetivo deste trabalho é desenvolver um modelo capaz de estimar automaticamente o volume de água de um reservatório, reduzindo a dependência de métodos manuais tradicionalmente utilizados por órgãos de monitoramento hídrico.

Como estudo de caso, foi utilizado o Açude de Orós, localizado no estado do Ceará, um dos principais reservatórios da região Nordeste.

### **Metodologia**

O modelo foi baseado na arquitetura U-Net, amplamente utilizada em tarefas de segmentação de imagens.

Etapas principais:
Coleta de imagens (2021–2025)
Pré-processamento dos dados
Treinamento da rede neural
Segmentação do espelho d'água
Estimativa do volume hídrico com base na área identificada

O dataset utilizado contém 30 imagens, contemplando diferentes níveis do reservatório — desde aproximadamente 19% da capacidade até sua sangria.

Resultados Apresentados

O modelo apresentou desempenho satisfatório tanto na segmentação quanto na estimativa de volume:

    Métricas de segmentação:
Coeficiente de Dice: 93,4%
Intersection over Union (IoU): 86,6%
    Métricas de estimativa de volume:
MAPE: 5,54%
RMSE: 6,99%
Bias: -0,14%
NSE: 0,950

Os resultados indicam alta precisão do modelo, com baixo erro em comparação aos dados oficiais.

🚀 Aplicações
Monitoramento hídrico automatizado
Apoio à gestão de recursos hídricos
Redução de custos operacionais
Escalabilidade para outros reservatórios
⚙️ Possíveis Melhorias
Aumento do volume de dados de treinamento
Uso de imagens com maior resolução
Integração com dados altimétricos
Ajustes finos na arquitetura da rede
📁 Estrutura do Projeto
├── data/              # Imagens e dados utilizados
├── preprocessing/     # Scripts de pré-processamento
├── model/             # Implementação da U-Net
├── training/          # Treinamento do modelo
├── evaluation/        # Avaliação e métricas
└── results/           # Resultados e outputs
🏷️ Palavras-chave
Processamento Digital de Imagens
Deep Learning
Sensoriamento Remoto
Monitoramento Hídrico
U-Net
📌 Conclusão

O modelo desenvolvido demonstrou ser uma ferramenta eficiente para estimativa de volume hídrico, com potencial aplicação prática. Com pequenas melhorias, pode atingir níveis ainda mais altos de precisão, tornando-se viável para uso em campo por órgãos responsáveis pelo monitoramento de reservatórios.