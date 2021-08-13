import streamlit as st
from transformers import pipeline
import bs4
import requests
import pdfplumber
from newspaper import Article
import torch
from transformers import Pipeline
from models.model_builder import ExtSummarizer
from ext_sum import summarize
import os
import nltk
import urllib.request
import string
from gtts import gTTS


def download_model():
    nltk.download('popular')
    url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)

        with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                                            (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


@st.cache(suppress_st_warning=True, show_spinner=False)
def load_model(model_type):
    checkpoint = torch.load(
        f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(
        device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model


@st.cache(suppress_st_warning=True, show_spinner=False, hash_funcs={Pipeline: id})
def abs_summary_model():
    return pipeline('summarization')


input_fp = "./raw_data/input.txt"
result_fp = './results/summary.txt'


def ext_summary(max_len):
    return summarize(input_fp, result_fp, model, max_length=max_len)


def show_text_summary_page():

    # Download model
    if not os.path.exists('checkpoints/mobilebert_ext.pt'):
        download_model()

    global model
    model = load_model('mobilebert')

    st.markdown("<h1 style='text-align: center;'>Automatic Text Summarizer</h3> <br> ",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>With one click, you can quickly summarize an article, blog, comprehension and more!</h2><br>""", unsafe_allow_html=True)

    global input_option
    input_option = st.radio(
        "Input Type:", ("Raw Text", "URL", "PDF"))

    if input_option == "Raw Text":
        st.markdown(
            "<h3 style='text-align: center;'>Input Raw Text</h3>", unsafe_allow_html=True)
        global TEXT_INPUT
        TEXT_INPUT = st.text_area(label="", height=300)

    elif input_option == "URL":
        st.markdown("<h3 style='text-align: center;'>Input URL</h3>",
                    unsafe_allow_html=True)
        global URL
        URL = st.text_input("")
        if URL != "":
            st.markdown(f"[*Click Here to Read Original Article*]({URL})")

    elif input_option == "PDF":
        st.markdown("<h3 style='text-align: center;'>Upload PDF</h3>",
                    unsafe_allow_html=True)
        global uploaded_file
        uploaded_file = st.file_uploader("", type="pdf")
        if uploaded_file is not None:
            st.text(" ")
            max_pages = len(pdfplumber.open(uploaded_file).pages)
            global page_range
            page_range = st.slider("Page Range:", min_value=1, max_value=max_pages, value=(
                1, 3 if max_pages >= 3 else max_pages))

    st.text("")
    global summary_type
    summary_type = st.radio("Summary Type:", ["Abstract", "Extract"])
    st.text(" ")

    global summary_len
    summary_len = st.radio("Summary Length:", ["Short", "Medium", "Long"])
    st.text(" ")

    global summary_display_format
    summary_display_format = st.radio("Summary Display Format:", [
                                      "Paragraph", "Bullet Points"])
    st.text(" ")

    global tts_enabled
    tts_enabled = st.checkbox("Text-To-Speech")
    st.write(" ")

    col1, col2, col3 = st.columns([1.2, 1, 1])
    make_summary = col2.button("Generate Summary")

    st.markdown("***")

    summary_for_tts = summary = ""
    if make_summary:
        with st.spinner(text="Generating Summary...."):
            msg = get_input_data()
            if msg[0] == "Error":
                st.error(msg[1])
            else:
                summary_for_tts, summary = generate_ext_summary(
                    msg[1])  # if summary_type == "Extract" else generate_abs_summary(msg[1])

    if summary != "":
        st.markdown("<h3 style='text-align: center;'>Summary</h3>",
                    unsafe_allow_html=True)
        st.markdown(f" {summary} ")

        if(tts_enabled):
            ta_tts = gTTS(summary_for_tts, lang='en', slow=False)
            ta_tts.save('tts_audio.mp3')
            audio_file = open('tts_audio.mp3', 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg', start_time=0)


def get_input_data():
    if input_option == "Raw Text":
        if TEXT_INPUT == "":
            return ["Error", "Text Missing"]
        else:
            TEXT = TEXT_INPUT
    elif input_option == "URL":
        if URL != "":
            TEXT = getArticleText(URL)
        else:
            return ["Error", "URL Missing"]

    elif input_option == "PDF":
        TEXT = ''

        if uploaded_file is not None:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in range(page_range[0]-1, page_range[1]):
                    TEXT += pdf.pages[page].extract_text()
        else:
            return ["Error", "PDF File Missing"]
    return ['Input data', TEXT]


def generate_abs_summary(TEXT):
    TEXT = TEXT.replace('.', '.<eos>')
    TEXT = TEXT.replace('!', '!<eos>')
    TEXT = TEXT.replace('?', '?<eos>')
    sentences = TEXT.split('<eos>')

    # maximum no. of words that can be summarized at a time
    max_block_size = 500

    # pointer
    current_block = 0

    # a block is a small part of the entire text containing full sentences
    # dont add one half of a sentence to one block and other half to another block
    blocks = []

    for sentence in sentences:
        # checks if we have a current block (a block with some text already added)
        if len(blocks) == current_block + 1:
            # checks if we can add another sentence to the current block
            if len(blocks[current_block]) + len(sentence.split(' ')) <= max_block_size:
                blocks[current_block].extend(sentence.split(' '))
            else:
                current_block += 1
                blocks.append(sentence.split(' '))
        else:
            blocks.append(sentence.split(' '))

    for index in range(len(blocks)):
        blocks[index] = ' '.join(blocks[index])

    min_len = 50
    max_len = 130
    if summary_len == "Medium":
        min_len = 100
        max_len = 180
    elif summary_len == "Long":
        min_len = 150
        max_len = 230

    try:
        model = abs_summary_model()
        summary_dict = model(TEXT, max_length=max_len,
                             min_length=min_len, do_sample=False)
    except IndexError:
        st.error("Error: unidentifiable text found (Possible cause: end of sentence not found). Instead, opt for EXTRACT type summary ")
    else:
        text_for_tts = final_summary = ' '.join(
            summary['summary_text'] for summary in summary_dict)
        if summary_display_format != "Paragraph":
            final_summary = '\n'.join(
                f"* {line}." if line != "" else "" for line in final_summary.split("."))

        return [text_for_tts, final_summary]


def generate_ext_summary(TEXT):

    # removing weird chars
    printable = set(string.printable)
    TEXT = ''.join(filter(lambda x: x in printable, TEXT))

    with open(input_fp, 'w', encoding="utf-8", errors="ignore") as file:
        file.write(TEXT)

    max_length = 3 if summary_len == "Short" else 5 if summary_len == "Medium" else 7
    text_for_tts = final_summary = ext_summary(max_length)
    if summary_display_format != "Paragraph":
        final_summary = '\n'.join(
            f"* {line}." if line != "" else "" for line in final_summary.split("."))

    return [text_for_tts, final_summary]


def getArticleText(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
