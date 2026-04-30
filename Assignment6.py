import streamlit as st
import re

st.title("Assignment6")
text = st.text_area("Enter something")

#test string:
"""
My name is jeff, i am going on a walk. Yesterday I walked five miles. 
I have been wanting to get in shape, so I have been making a point to exercise regularly. 
"""

def processtxt(text):    
    #Keeping only Text and digits
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    #Removes Whitespaces
    text = re.sub(r"\'s", " ", text)
    # Removing Links if any
    text = re.sub(r"http\S+", " link ", text)
    # Removes Punctuations and Numbers
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
    # Splitting Text
    text = text.split()
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lematized_words)
    st.write(text)


if st.button('Analyze'): # process when clicked
    processtxt(text)
