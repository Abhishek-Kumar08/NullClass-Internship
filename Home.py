import streamlit as st

c1,c2,c3 = st.columns(3)
c2.title('NULLCLASS')
c1,c2,c3 = st.columns([1,4,1])
c2.subheader('NULLCLASS EDTECH PRIVATE LIMITED')
c2.write('\n')
c1,c2,c3 = st.columns([1,7,1])
c2.header('Machine Learning Developer Intern')
c2.write('\n')
c1,c2,c3 = st.columns([1,20,1])
c2.subheader('Introduction:')
c2.write('\n')
c2.markdown('''I'm Abhishek Kumar, currently in my 8th semester pursuing BTech in Computer Science and Engineering with a focus on Artificial Intelligence at Parul University, Vadodara, Gujarat. My academic journey has been both challenging and exciting, allowing me to explore the vast realms of technology.
I'm passionate about the applications of AI and its potential to address societal challenges. Beyond the classroom, I actively participate in projects, aiming to bridge the gap between theory and real-world implementation.
''')
c2.markdown('This webapp contains all of my projects completed during my internship period!')
