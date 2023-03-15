import streamlit as st

st.write("Interpret the sentiment of this given text")

st.textarea(label, value="prompt", height=300,max_chars=450, placeholder = "Today is a good day, Sentiment: positive")

if st.button(label="Interpret", type = "primary"):
    pass

