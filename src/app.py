import tensorflow as tf
import numpy as np
import plotly.express as px
import streamlit as st
import io

st.markdown("<h1 style='text-align: left; color: #00eeff;'>animal classifier (Demonstration)</h1>", unsafe_allow_html=True)

classes = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}

option_weights = st.selectbox(
    'Which model Weights would you like to test with?',
    ('current','legacy', 'custom'))

st.write(f'You selected: {option_weights} Weights')

custom_weight = False
match option_weights:
    case 'current':
        weight_path = 'src\Model\weights_animals_resnet.h5'
        custom = False
    case 'legacy':
        weight_path = 'src\Model\weights_animals_resnet(old).h5'
        custom = False
    case 'custom':
        st.write('IN DEVELOPMENT!')
        # custom_weight = True
        
# if custom_weight:
#     uploaded_weight = st.file_uploader("Upload Weights", type="h5")
#     if uploaded_weight is not None:
#         weight_bytes = uploaded_weight.read()
#         weights = io.BytesIO(weight_bytes)
#         file_details = {"filename":uploaded_weight.name, "filetype":uploaded_weight.type, "filesize":uploaded_weight.size}
#         st.write(file_details)
#     weight_path = weights



option = st.selectbox(
    'Which animal would you like to test with?',
    ('Cat', 'Dog', 'Elephant', 'horse', 'lion', 'custom'))

st.write(f'You selected: {option} animal')

custom = False
match option:
    case 'Cat':
        animal_path = 'src\Data\inf\cat.jpg'
        custom = False
    case 'Dog':
        animal_path = 'src\Data\inf\dog.jpg'
        custom = False
    case 'Elephant':
        animal_path = 'src\Data\inf\elephant.jpg'
        custom = False
    case 'horse':
        animal_path = 'src\Data\inf\horse.jpg'
        custom = False
    case 'lion':
        animal_path = 'src\Data\inf\lion.jpg'
        custom = False
    case 'custom':
        custom = True

if custom:
    uploaded_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)
    animal_path = uploaded_file

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
    model.load_weights(weight_path)
    return model

with st.spinner("Loading Model...."):
    model = loading_model()


with st.spinner("predicting the image...."):
    prediction = model.predict(image_data).argmax(axis=1)
    class_name = list({k for k in classes if classes[k]==prediction})[0]

    st.markdown("<h2 style='text-align: center; color: white;'>Predicted class name: </h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: green;'>{}</h3>".format(class_name), unsafe_allow_html=True)
