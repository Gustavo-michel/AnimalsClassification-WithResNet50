# Classificador de Animais com ResNet50

Este projeto utiliza a arquitetura de rede neural ResNet50 para classificar imagens de animais. 

## Dataset

O dataset utilizado neste projeto está disponível no Kaggle. Ele contém diversas imagens de animais, separados por 5 classes que são usadas para treinar e testar o modelo de classificação. Você pode acessar o dataset aqui. [Animals](https://www.kaggle.com/datasets/antobenedetti/animals/data)

## Modelo e Transfer Learning

O modelo de classificação é baseado na arquitetura ResNet50, uma rede neural convolucional profunda. Este projeto utiliza a técnica de Transfer Learning, que consiste em usar um modelo pré-treinado (neste caso, a ResNet50 treinada no ImageNet) e adaptá-lo para um novo problema. Isso permite aproveitar os recursos aprendidos pelo modelo em tarefas anteriores que são relevantes para a tarefa atual.

## Carregando o Modelo

Para carregar o modelo treinado, você pode usar a função `load_model` do Keras. Aqui está um exemplo de como fazer isso:

```python
from keras.models import load_model

# Carrega o modelo
model = load_model('caminho/para/o/classificador_animals_Resnet.sav')

ou

with open('caminho/para/o/classificador_animals_Resnet.sav', "rb") as m:
    model = pickle.load(m)
```
