import tensorflow as tf
import numpy as np
import plotly.express as px
import streamlit as st

st.markdown("<h1 style='text-align: left; color: #00eeff;'>animal classifier (Demonstration)</h1>", unsafe_allow_html=True)

classes = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}

option = st.selectbox(
    'How would aimal do you like test?',
    ('Cat', 'Dog', 'Elephant', 'horse', 'lion'))

st.write('You selected:', option)

match option:
    case 'Cat':
        animal_path = 'src\Data\inf\cat.jpg'
    case 'Dog':
        animal_path = 'src\Data\inf\dog.jpg'
    case 'Elephant':
        animal_path = 'src\Data\inf\elephant.jpg'
    case 'horse':
        animal_path = 'src\Data\inf\horse.jpg'
    case 'lion':
        animal_path = 'src\Data\inf\lion.jpg'

def preprocessing_images(data):
    data = tf.keras.preprocessing.image.load_img(data, target_size = (224, 224))
    data = tf.keras.preprocessing.image.img_to_array(data)
    data = np.expand_dims(data, axis = 0)
    data = tf.keras.applications.resnet50.preprocess_input(data)
    return data

image_data = preprocessing_images(animal_path)

def plotImage(image):
    image = tf.keras.preprocessing.image.load_img(image)
    st.markdown("<h2 style='text-align: left; color: white;'>Animal Plot: </h2>", unsafe_allow_html=True)
    image = px.imshow(image)
    return image

plot = plotImage(animal_path)
st.plotly_chart(plot)


def create_model():
    base_model = tf.keras.applications.ResNet50(weights = 'imagenet', include_top = False)

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation= 'relu')(x)
    x = tf.keras.layers.Dense(1024, activation= 'relu')(x)
    x = tf.keras.layers.Dense(1024, activation= 'relu')(x)
    x = tf.keras.layers.Dense(512, activation= 'relu')(x)
    preds = tf.keras.layers.Dense(5, activation= 'softmax')(x)

    model = tf.keras.models.Model(inputs = base_model.input, outputs = preds)
    model.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# @st.cache(allow_output_mutation=True)
def loading_model():
    model = create_model()
    model.load_weights("src/Model/animals_resnet_weights.h5")
    return model

with st.spinner("Loading Model...."):
    model = loading_model()



prediction = model.predict(image_data).argmax(axis=1)
class_name = list({k for k in classes if classes[k]==prediction})[0]

st.markdown("<h2 style='text-align: center; color: white;'>Predicted class name: </h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: green;'>{}</h3>".format(class_name), unsafe_allow_html=True)
