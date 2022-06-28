#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import cv2
from tensorflow import keras
import numpy as np
import os
import numpy


# In[4]:


def about():
    st.write('''**Recommend songs by detecting facial emotion''')


# In[5]:


def main():
    st.title("Recommend songs by detecting facial emotion ")
   # st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = keras.models.load_model('model.h5')

    if choice == "Home":
        
        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # upload
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        
        if image_file is not None:
            
            #img = cv2.imread(image_file)
            img = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            
            if st.button("Process"):
                
                frame = cv2.resize(img,(48,48),interpolation=cv2.INTER_BITS2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
                
                gray = gray.reshape(1,48,48,1)
                
                predicts = model.predict(gray)[0]
                label = EMOTIONS[predicts.argmax()]
                st.write('Detected emotion is', label)
                
                st.write("## Recommended audio")
                if label == 'angry':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label == 'disgust':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label == 'fear':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label == 'happy':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label == 'neutral':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label == 'sad':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    
                elif label ==  'surprise':
                    st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
                    

    elif choice == "About":
        about()


# In[6]:


if __name__ == "__main__":
    main()


# In[ ]:




