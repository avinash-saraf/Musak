# Musak
Musak is an automatic text summarization and question-answering tool. UI built using Streamlit.
Auto text-summarisation has two modes: 
  - Abstract: Performs abstractive summarization using the summarization pipeline by HuggingFace (uses BART model)
  - Extract: Performs extractive summarization using the pre-trained DistilBERT Model (trained and fine-tuned by https://github.com/chriskhanhtran/bert-extractive-summarization)

Question-answering using BERT-large trained by SQuAD dataset (specifically for question answering purposes)


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
