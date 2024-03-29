# Classificador de Animais com ResNet50

Este projeto utiliza a arquitetura de rede neural ResNet50 para classificar imagens de animais. 

## Dataset

O dataset utilizado neste projeto está disponível no Kaggle. Ele contém diversas imagens de animais, separados por 5 classes que são usadas para treinar e testar o modelo de classificação. Você pode acessar o dataset aqui. [Animals](https://www.kaggle.com/datasets/antobenedetti/animals/data)

## Modelo e Transfer Learning

O modelo de classificação é baseado na arquitetura ResNet50, uma rede neural convolucional profunda. Este projeto utiliza a técnica de Transfer Learning, que consiste em usar um modelo pré-treinado (neste caso, a ResNet50 treinada no ImageNet) e adaptá-lo para um novo problema. Isso permite aproveitar os recursos aprendidos pelo modelo em tarefas anteriores que são relevantes para a tarefa atual.

## Carregando o Modelo

Para carregar o modelo treinado, você pode usar a função `load_model` do Keras. Aqui está um exemplo de como fazer isso:

```python
*Criação da base do modelo e compilção

# Carrega os pesos já treinadados do modelo
model = load_weights('caminho/para/o/weights_animals_Resnet.h5')
```

## Acess the app
In terminal:
```
streamlit run https://github.com/Gustavo-michel/AnimalsClassification-WithResNet50/blob/main/src/app.py
```

## Preview
![Captura de tela 2024-02-28 224039](https://github.com/Gustavo-michel/AnimalsClassification-WithResNet50/assets/127684360/f34ccb71-bbb8-4263-a27e-0afcf4650ada)

![Captura de tela 2024-02-28 224105](https://github.com/Gustavo-michel/AnimalsClassification-WithResNet50/assets/127684360/70d9a3de-a6fc-4399-b96c-2e72e1116a8c)
