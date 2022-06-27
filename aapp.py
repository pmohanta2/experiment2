#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import cv2
from tensorflow import keras
import numpy as np
import os


# In[4]:


def about():
    st.write('''**Haar Cascade** is an object detection algorithm.
        It can be used to detect objects in images or videos. 
        The algorithm has four stages:
                1. Haar Feature Selection 
                2. Creating  Integral Images
                3. Adaboost Training
                4. Cascading Classifiers
Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid''')


# In[5]:


def main():
    st.title("Face Detection App ")
   # st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    
    EMOTIONS = ['Angry', 'Disgust', 'Happy', 'Sad', 'Surprise', 'Neutral']
    model = keras.models.load_model('model.h5')

    if choice == "Home":
        
        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        
        if image_file is not None:
            
            img = cv.imread(image_file)
            
            if st.button("Process"):
                
                frame = cv2.resize(img,(48,48),interpolation=cv2.INTER_BITS2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
                
                gray = gray.reshape(1,48,48,1)
                
                predicts = model.predict(gray)[0]
                label = EMOTIONS[predicts.argmax()]
                st.write('Detected emotion is', label)

    elif choice == "About":
        about()


# In[6]:


if __name__ == "__main__":
    main()


# In[ ]:




