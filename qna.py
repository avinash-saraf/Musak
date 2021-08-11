import streamlit as st 
import requests
import bs4
import pdfplumber
import torch 
from transformers import BertForQuestionAnswering, BertTokenizer
from newspaper import Article

def show_qna_page():

    st.markdown("<h1 style='text-align: center;'>Automatic Question Answering</h3> <br> ",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>With one keypress, you can quickly find the answer to a question regarding a passage of text!</h2><br>""",unsafe_allow_html=True)

    global input_option
    input_option = st.select_slider("Input Type:", ("Raw Text","URL","PDF"), value="URL")
 
    if input_option == "Raw Text":
        st.markdown("<h3 style='text-align: center;'>Input Raw Text</h3>",unsafe_allow_html=True)
        global TEXT_INPUT
        TEXT_INPUT = st.text_area(label="", height=300)

    elif input_option == "URL":
        st.markdown("<h3 style='text-align: center;'>Input URL</h3>",unsafe_allow_html=True)
        global URL
        URL = st.text_input("")
        if URL != "":
            st.markdown(f"[*Click Here to Read Original Article*]({URL})")
        st.text(" ")
        global news_article
        news_article = st.checkbox("Newspaper article")
        
    elif input_option == "PDF":
        st.markdown("<h3 style='text-align: center;'>Upload PDF</h3>",unsafe_allow_html=True)
        global uploaded_file
        uploaded_file = st.file_uploader("",type="pdf")
        if uploaded_file is not None:
            st.text(" ")
            max_pages = len(pdfplumber.open(uploaded_file).pages)
            global page_range
            page_range = st.slider("Page Range:",min_value=1, max_value=max_pages,value=(1, 3 if max_pages >=3 else max_pages))

    st.write(" ")
    st.markdown("<h3 style='text-align: center;'>Ask A Question</h3>",unsafe_allow_html=True)
    q = st.text_input(" ")
    st.write(" ")

    if q!="":
        with st.spinner(text="Answering Question...."):
            msg = answer(q)
            if msg != None:
                st.error(msg)


def answer(q):

    if input_option == "Raw Text":
        if TEXT_INPUT == "":
            return "Text Missing"
        else:
            TEXT = TEXT_INPUT
    elif input_option == "URL":
        if URL != "":
            if news_article:
                TEXT = getArticleText(URL)
            else:
                r = requests.get(URL)
                soup = bs4.BeautifulSoup(r.text, 'html.parser')
                results = soup.find_all(['h1', 'p'])
                text = [result.text for result in results]
                TEXT = ' '.join(text)
        else:
            return "URL Missing"

    elif input_option == "PDF":
        TEXT = ''

        if uploaded_file is not None:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in range(page_range[0]-1, page_range[1]):
                    TEXT += pdf.pages[page].extract_text()
        else:
            return "PDF File Missing"

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    answer_text = TEXT + " "

    #limiting number of words
    max_words = 500

    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(q, answer_text)[0:max_words]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)[0:max_words]

    ##Segmenting the input ids into segement A (to indicate its the question) and segment B (to indicate its the answer)

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # Run our example through the BERT model.
    outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                                return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
        
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer
    answer = ' '.join(tokens[answer_start:answer_end+1])

    ## Cleaning the text to join sub-words
    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    
    if answer == "[CLS]":
        st.error("Question cannot be answered")
        return 
    st.markdown(f" __Answer:__ {answer} ")




def getArticleText(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text