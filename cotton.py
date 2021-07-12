import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

st.title('COTTON LEAVES DISEASE DETECTION')
st.write('This is a cotton disease detection web app using streamlit.')
st.text('upload an image')
model=pickle.load(open('cotton.pkl','rb'))

uploaded_file=st.file_uploader('choose an image',type='jpg')
if uploaded_file is not None:
  img=imread(uploaded_file)
  st.image(img,caption='uploaded image')
  if st.button('PREDICT'):
    CATAGORIES=['diseased cotton leaf','fresh cotton leaf']
    st.write('Results')
    flat_data=[]
    img=np.array(img)
    img_resized=resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    q=model.predict_proba(flat_data)
    for index,item in enumerate(CATAGORIES):
       st.write(f'{item} : {q[0][index]*100}')
    y_out=CATAGORIES[y_out[0]]
    st.title(y_out)
st.write('HOW TO USE')
st.write('step1.first click on the option browse files,it will open the camera and files in your mobile')
st.write('step2.choose any of the option camera or files and take a picture of your face')
st.write('step3.after loading the image,click on predict button then it will predict that your cotton plant is diseased or not')
st.write('It is made by using Transfer Learning technique')
st.write("Transfer Learning is the reuse of a pre trained model on a new problem.It's currently very popular in deep learning because it can train deep neural network with comparatively little data.This is very useful in data science field since most of real world problems do not have millions of labelled data")
st.write('Accuracy of this model is :85%')
st.write('For more queries please contact:shekharboppanapally944@gmail.com')    