import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import mediapipe as mp
from keras.models import load_model

model = load_model('SignLanguage_Detector.h5')

flag = True

st.title('Sign Language Detection')
st.markdown('This machine learning model can identify and recognize the American Sign Language alphabets from A to Y from any video or image.')
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
    im = cv2.resize(im,(28,28))
    im = im/255
    im = np.expand_dims(im,axis=0)
    e = model.predict(im)
    sign = chr(65+e[0].argmax())
    if st.button('Detect Image'):
        result = f"Dtected alphabet is: {sign}"
        st.success(result)

def pred_video(file_path):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

    cap = cv2.VideoCapture(file_path) 

    if (cap.isOpened()== False): 
        print("Error opening video file") 

    stframe = st.empty()

    while(cap.isOpened()): 
        
        x_ = []
        y_ = []
        ret, frame = cap.read() 
        if ret == True: 
            H,W,_ = frame.shape
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
            
                x1 = int(min(x_)*W)-30
                y1 = int(min(y_)*H)-30
                x2 = int(max(x_)*W)+30
                y2 = int(max(y_)*H)+30

                im = im[y1:y2,x1:x2]
                im = cv2.resize(im,(28,28))
                im = im/255
                im = np.expand_dims(im,axis=0)
                pred = model.predict(im)
                pred = chr(65+pred[0].argmax())

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,100,200),2)
                cv2.putText(frame,pred,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,100,200),2,cv2.LINE_AA)

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
