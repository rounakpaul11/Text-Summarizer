# Importing required libraries
import streamlit as st
import nltk
import os

# Setting NLTK data path
nltk.data.path.append(os.path.join(os.path.dirname(nltk.__file__), 'nltk_data'))

# Download NLTK data
nltk.download('stopwords', download_dir='./nltk_data/')
nltk.download('punkt', download_dir='./nltk_data/')
nltk.download('wordnet', download_dir='./nltk_data/')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
import language_tool_python
from nltk.corpus import wordnet

# Streamlit app
def main():
    st.title("Text Analyzer App")
    
    # Setting up the grammar tool
    my_tool = language_tool_python.LanguageToolPublicAPI('en-US')

    # Input text for analysis
    text = st.text_area("Enter the text:", "")

    if st.button("Analyze"):
        # Correct grammatical mistakes
        corrected_text = correct_grammar(text, my_tool)
        st.subheader("Grammatically Corrected Text:")
        st.write(corrected_text)

        # Summarize the text
        summary_text = summarize_text(corrected_text)
        st.subheader("Summary:")
        st.write(summary_text)

        # Find synonyms of a word
        word = st.text_input("Enter a word to find synonyms:", "")
        synonyms = find_synonyms(word)
        st.subheader(f"Synonyms of the word '{word}' are:")
        st.write(", ".join(synonyms))

# Function to correct grammatical mistakes
def correct_grammar(text, my_tool):
    matches = my_tool.check(text)
    corrected_text = my_tool.correct(text)
    return corrected_text

# Function to summarize the text
def summarize_text(text):
    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    # Creating a frequency table
    freqTable = {}
    for word in words:
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a sentence value dictionary
    sentences = sent_tokenize(text)
    sentenceValue = {}

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    # Calculate average sentence value
    sumValues = sum(sentenceValue.values())
    average = int(sumValues / len(sentenceValue))

    # Generate summary
    summary = ""
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary

# Function to find synonyms of a word
def find_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

if __name__ == "__main__":
    main()
