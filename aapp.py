{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9d6adb14",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import cv2\n",
    "import keras\n",
    "import numpy as np\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "19ab2068",
   "metadata": {},
   "outputs": [],
   "source": [
    "def about():\n",
    "    st.write('''**Haar Cascade** is an object detection algorithm.\n",
    "        It can be used to detect objects in images or videos. \n",
    "        The algorithm has four stages:\n",
    "                1. Haar Feature Selection \n",
    "                2. Creating  Integral Images\n",
    "                3. Adaboost Training\n",
    "                4. Cascading Classifiers\n",
    "Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html\n",
    "https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid''')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5da67f31",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    st.title(\"Face Detection App \")\n",
    "   # st.write(\"**Using the Haar cascade Classifiers**\")\n",
    "\n",
    "    activities = [\"Home\", \"About\"]\n",
    "    choice = st.sidebar.selectbox(\"Pick something fun\", activities)\n",
    "    \n",
    "    EMOTIONS = ['Angry', 'Disgust', 'Happy', 'Sad', 'Surprise', 'Neutral']\n",
    "    model = keras.models.load_model('model.h5')\n",
    "\n",
    "    if choice == \"Home\":\n",
    "        \n",
    "        st.write(\"Go to the About section from the sidebar to learn more about it.\")\n",
    "        \n",
    "        # You can specify more file types below if you want\n",
    "        image_file = st.file_uploader(\"Upload image\", type=['jpeg', 'png', 'jpg', 'webp'])\n",
    "        \n",
    "        if image_file is not None:\n",
    "            \n",
    "            img = cv.imread(image_file)\n",
    "            \n",
    "            if st.button(\"Process\"):\n",
    "                \n",
    "                frame = cv2.resize(img,(48,48),interpolation=cv2.INTER_BITS2)\n",
    "                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0\n",
    "                \n",
    "                gray = gray.reshape(1,48,48,1)\n",
    "                \n",
    "                predicts = model.predict(gray)[0]\n",
    "                label = EMOTIONS[predicts.argmax()]\n",
    "                st.write('Detected emotion is', label)\n",
    "\n",
    "    elif choice == \"About\":\n",
    "        about()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5bdb080c",
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0dc600fd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
