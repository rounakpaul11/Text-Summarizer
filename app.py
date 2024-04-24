# Importing required libraries
import streamlit as st
import nltk
import os
import random
from transformers import BartForConditionalGeneration, BartTokenizer, GPT2Tokenizer, GPT2LMHeadModel, TransformerSummarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import language_tool_python
from nltk.corpus import wordnet

# Set NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download NLTK data
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# BERT Summarizer
bert_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bert_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# GPT-2 Summarizer
GPT2_model = TransformerSummarizer(transformer_type="GPT2", transformer_model_key="gpt2-medium")

# Ensure WordNet is loaded
try:
    wordnet.ensure_loaded()
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

# Function to correct grammatical mistakes
def correct_grammar(text, my_tool):
    matches = my_tool.check(text)
    corrected_text = my_tool.correct(text)
    return corrected_text

# Function to summarize the text using NLTK
def summarize_text(text, threshold):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    freqTable = {}
    
    for word in words:
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = {}

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = sum(sentenceValue.values())
    average = int(sumValues / len(sentenceValue))

    summary = ""
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (threshold * average)):
            summary += " " + sentence

    return summary

# Function to summarize text using BERT
def bert_summarize(text):
    inputs = bert_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bert_model.generate(inputs.input_ids, num_beams=4, min_length=60, max_length=200, early_stopping=True)
    bert_summary = bert_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return bert_summary

# Function to summarize text using GPT-2
def gpt2_summarize(text):
    full = ''.join(GPT2_model(text, min_length=60))
    return full

# Function to find a random synonym of a word from WordNet
def find_random_synonym(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    
    unique_synonyms = list(set(synonyms))
    if unique_synonyms:
        return random.choice(unique_synonyms)
    else:
        return "No synonyms found"

# Streamlit app
def main():
    st.markdown("<h1>Text Analyzer App</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        %s
    </style>
""" % open("style.css").read(), unsafe_allow_html=True)
    
    my_tool = language_tool_python.LanguageToolPublicAPI('en-US')
    
    text = st.text_area("Enter the text:", "")
    
    st.sidebar.title("Summary Length")
    threshold = st.sidebar.slider("Select threshold for summary length:", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

    if st.button("Analyze"):
        corrected_text = correct_grammar(text, my_tool)
        st.subheader("Grammatically Corrected Text:")
        st.write(corrected_text)

        nltk_summary = summarize_text(corrected_text, threshold)
        st.subheader("NLTK Summary:")
        st.write(nltk_summary)

        bert_summary = bert_summarize(corrected_text)
        st.subheader("BERT Summary:")
        st.write(bert_summary)

        gpt2_summary = gpt2_summarize(corrected_text)
        st.subheader("GPT-2 Summary:")
        st.write(gpt2_summary)

    st.write("Made by Rounak Paul(21bce1566) & Tushar Panwar(21bce1074)")

    st.sidebar.title("Synonym Finder")
    input_word = st.sidebar.text_input("Enter a word to find its synonym:", "")

    if st.sidebar.button("Find Synonym"):
        if input_word:
            synonym = find_random_synonym(input_word)
            st.sidebar.subheader(f"A random synonym of '{input_word}' is:")
            st.sidebar.write(synonym if synonym != "No synonyms found" else "No synonyms found")
        else:
            st.sidebar.warning("Please enter a word.")

if __name__ == "__main__":
    main()
