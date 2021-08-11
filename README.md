# Musak
Musak is an automatic text summarization and question-answering tool. UI built using Streamlit. <br>
Its auto text-summarisation feature has two modes: 
  - Abstract: Performs abstractive summarization using the summarization pipeline by HuggingFace (uses BART model). 
  - Extract: Performs extractive summarization using the pre-trained DistilBERT Model (trained and fine-tuned by https://github.com/chriskhanhtran/bert-extractive-summarization)

It allows the user to select between both types of summarization techniques because one may perform better over the other for certain types of input text. 

It also answers questions using BERT-large trained by SQuAD dataset (specifically for question answering purposes).


## Setup 
 - Clone the repository using git clone https://github.com/avinash-saraf/Musak.git
 - Once inside the project directory, (assuming you have a working Python > 3.8 environment), run pip install -r requirements.txt
 - Run the webapp using streamlit run app.py

  ```
  git clone https://github.com/avinash-saraf/Musak.git
  cd Musak
  pip install -r requirements.txt
  streamlit run app.py
  ```
