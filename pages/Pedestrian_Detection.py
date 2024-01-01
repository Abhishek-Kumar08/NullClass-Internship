import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from keras.models import load_model

model = load_model('Pedestrian_Detector.h5')

flag = True

st.title('Pedestrian Detection')
st.markdown('This machine learning model can if the image contains any pedestrian or not. It can also draw rectangles around the pedestrians in any video.')

st.write('\n')

st.subheader('Image Detection')
st.write('\n')

a = st.file_uploader("Upload any image")

st.write('\n')

if a:
    st.write("Filename:", a.name)
    st.image(a)
    im = Image.open(a)
    im = im.save("img.jpg")
    im = cv2.imread('img.jpg',0)
    im = cv2.resize(im,(64,64))
    im = im/255 
    im = np.expand_dims(im,axis=0)
    pred = model.predict(im)
    pred = 'Pedestrian' if pred[0]>0.5 else 'No Pedestrian'
    if st.button('Detect Image'):
        result = f"{pred}s Detected!"
        st.success(result)

def pred_video(file_path):
    cap = cv2.VideoCapture(file_path)

    pedestrian_classifier = 'pedestrian.xml'

    pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

    stframe = st.empty()

    while flag:
        (ret, frame) = cap.read()

        if ret:
            frame = cv2.resize(frame,(1080,720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break

        pedestrians = pedestrian_tracker.detectMultiScale(gray_frame,1.1,9)

        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Pedestrian', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        stframe.image(frame)

    cap.release()
    
st.write('\n')

st.subheader('Video Detection')
st.write('\n')

v = st.file_uploader("Upload any video")

st.write('\n')

if v:
    st.write("Filename:", v.name)
    st.video(v)
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(v.read())
    if st.button('Detect Video'):
        if st.button('Stop'):
            flag = False
        pred_video(tfile.name)
