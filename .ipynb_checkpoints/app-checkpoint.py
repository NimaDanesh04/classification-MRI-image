import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2 as cv

st.markdown('<style>body {background-color: #f0f0f0;}</style>', unsafe_allow_html=True)
st.markdown('<h2 style="color: red; text-align: center;">"Diagnosing tumors with just one click"</h2>', unsafe_allow_html=True)

model = load_model('wheight/model.h5')

input_image = st.file_uploader("Upload Your Image")

if input_image is not None:
    image = np.array(Image.open(input_image))
    image1 = cv.resize(image, (128, 128))
    fainal_image = image1.reshape(1, 128, 128, 3)
    predict = model.predict(fainal_image)
    result = np.argmax(predict, axis=1)
    if result == 0:
        st.error('tumor can be seen in this photo')
        st.image(image)
    elif result == 1:
        st.success('No tumor can be seen in this picture')
        st.image(image)

st.write('If you want to see the notebook of this model, click on this link')
st.markdown('[link](https://www.kaggle.com/code/nimadanesh/tumor-diagnosis-with-cnn-model)')
