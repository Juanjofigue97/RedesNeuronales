from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np

import tensorflow as tf
import streamlit as st
from PIL import Image
from skimage.transform import resize

tf.function(experimental_relax_shapes=False)

# Modelos entrenados
MODELO1 = 'LaPoderosa2.h5'
MODELO2 = 'SOYLEYENDA.h5'

# Dimensiones de las imagenes de entrada    
width_shape = 64
height_shape = 64

# Clases
names = ['0','1','2','3','4','5']


def model_prediction(img, model):

    img_resize = resize(img, (width_shape, height_shape))
    x=preprocess_input(img_resize*255)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds

st.title(" CLASIFICADOR DE MANOS")
IA_name = st.sidebar.selectbox("Selecciona Inteligencia Artificial",("La poderosa","SOY LEYENDA"))

def main():
    
    if IA_name == 'La poderosa':
        model=''

        # Se carga el modelo
        if model=='':
            model = load_model(MODELO1)
        
        predictS=""
        st.write("LA PODEROSA")
        img_file_buffer = st.file_uploader("Carga una imagen ", type=["png", "jpg", "jpeg"])
        
        # El usuario carga una imagen
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))    
            st.image(image, caption="Imagen", use_column_width=False)
        
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción"):
            predictS = model_prediction(image, model)
            st.success('EL NUMERO ES: {}'.format(names[np.argmax(predictS)]))
    if IA_name == 'SOY LEYENDA':
        model=''

        # Se carga el modelo
        if model=='':
            model = load_model(MODELO2)
        
        predictS=""
        st.write("SOY LEYENDA")
        img_file_buffer = st.file_uploader("Subir la imagen a clasificar", type=["png", "jpg", "jpeg"])
        
        # El usuario carga una imagen
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))    
            st.image(image, caption="Imagen", use_column_width=False)
        
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predicción"):
            predictS = model_prediction(image, model)
            st.success('EL NUMERO ES: {}'.format(names[np.argmax(predictS)]))
        

if __name__ == '__main__':
    main()