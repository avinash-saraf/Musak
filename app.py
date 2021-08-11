import streamlit as st 
from text_summary import show_text_summary_page
from home import show_home_page
from qna import show_qna_page

page = st.sidebar.selectbox("Select Page", ("Home", "Summarizer", "Question-Answering"))

if page == "Home":
    show_home_page()
elif page == "Summarizer":
    show_text_summary_page()
elif page == "Question-Answering":
    show_qna_page()


