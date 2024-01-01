import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from keras.models import load_model

model = load_model('Human_Activity_Recognition.h5')

flag = True

activity = ['Calling', 'Clapping', 'Cycling', 'Dancing', 'Drinking', 'Eating','Fighting', 'Hugging', 'Laughing', 'Listening to music', 'Running','Sitting', 'Sleeping', 'Texting', 'Using laptop']

st.title('Human Activity Recognition')
st.markdown('This machine learning model can identify and recognize some of the basic human activities that might be happening in any video or images.')
st.markdown('The complete list of activities that it can recognize are listed below:')
a1,a2,a3,a4,a5 = st.columns(5)
for i in range(3):
    a1.markdown(activity[5*i])
    a2.markdown(activity[5*i+1])
    a3.markdown(activity[5*i+2])
    a4.markdown(activity[5*i+3])
    a5.markdown(activity[5*i+4])

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
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(48,48))
    im = im/255
    image = []
    for i in range(21):
        image.append(im)
    image = np.array(image)
    e = model.predict(image)
    e = e[0].argmax()
    act = activity[e]
    if st.button('Detect Image'):
        result = f"Predicted activity is: {act}"
        st.success(result)

def pred_video(file_path):
    cap = cv2.VideoCapture(file_path) 

    if (cap.isOpened()== False): 
        print("Error opening video file") 

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (100, 100) 
    fontScale = 1.5
    color = (255, 150, 50) 
    thickness = 2

    stframe = st.empty()

    while(cap.isOpened()): 
        
        ret, frame = cap.read() 
        if ret == True: 
            frame = cv2.resize(frame, (1080,720))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            im = cv2.resize(frame,(48,48))
            im = im/255
            image = []
            for i in range(21):
                image.append(im)
            image = np.array(image)
            e = model.predict(image)
            e = e[0].argmax()
            e = activity[e]
            frame = cv2.putText(frame, e, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    
        else: 
            break

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
