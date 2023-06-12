import streamlit as st
from svm_deploy import load_system, test_model

st.write("Sentiment Analysis")

text = st.text_area("Interpret the sentiment of this given text", value=" ", height=200, max_chars=450, placeholder = "Today is a good day, Sentiment: positive")

if st.button(label="Interpret", type = "secondary"):
    result = test_model(text)
    st.write(result)

