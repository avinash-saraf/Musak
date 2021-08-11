import streamlit as st 


def show_home_page():

    name = "Musak"

    st.markdown(f"<h1 style='text-align:center;'>Welcome to {name}! </h1>",unsafe_allow_html=True)

    st.markdown(f"<h2 style='text-align:center;'>{name} is a quick automatic text summarizer and question-answering tool." \
    " You can easily paraphrase articles, blogs, and/or comprehensions with the click of a button!"\
        " You can also generate answers to simple factual questions about these texts.</h2>",unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;'> Select the \'Summarizer\' page "\
    "or \'Question-Answering\' page from the sidebar to use these features. </h2>", unsafe_allow_html=True)
    