import pickle
import tensorflow as tf
import numpy as np
import plotly.express as px
import streamlit as st
st.config.set_option("server.maxUploadSize", 800)


classes = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}

def preprocessing(data):
    data = tf.keras.preprocessing.image.load_img(data, target_size = (224, 224))
    data = tf.keras.preprocessing.image.img_to_array(data)
    data = np.expand_dims(data, axis = 0)
    data = tf.keras.applications.resnet50.preprocess_input(data)
    return data

def plotImage(image):
    image = tf.keras.preprocessing.image.load_img(image, target_size = (224, 224))
    return px.imshow(image)


image = preprocessing('test\Data\inf\cat.jpg')
imagePlot = plotImage('test\Data\inf\cat.jpg')

st.plotly_chart(imagePlot)

with open("test\Model\classificador_animals_Resnet.sav", 'rb') as m:
    model = pickle.load(m)


# prediction = model.predict(image).argmax()
# class_name = list({k for k in classes if classes[k]==prediction})[0]