import streamlit as st
import librosa as rosa
import numpy as np
import cv2
from keras.models import load_model

model = load_model('D:\Productive\My_Programs\Visual Studio Code\Python Programming\NULLCLASS ML INTERNSHIP\Streamlit Web App\Voice_Gender_Detector.h5')

st.title('Voice Gender Detection')
st.markdown('This machine learning model can detect the gender of the person on the basis of their voices. To use the model simply upload any .mp3 audio file without any background noise and the model with detct your gender.')

st.write('\n')

a = st.file_uploader("Upload any .mp3 file")

st.write('\n')

if a:
    st.write("Filename:", a.name)
    st.audio(a)
    
    if st.button('Detect'):
        y, sr = rosa.load(a, sr=None)
        d = rosa.stft(y)
        db = rosa.amplitude_to_db(np.abs(d), ref=np.max)
        db = cv2.resize(db,(64,64))
        db = (db+80)/80
        img = []
        for i in range(20):
            img.append(db)
        img = np.array(img)
        e = model.predict(img)
        gen = 'Male' if e[0]<0.5 else 'Female'
        result = f"Predicted gender is: '{gen}'"
        st.success(result)



