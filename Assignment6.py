import streamlit as st
# NLP Libraries
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import re
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Assignment6")
text = st.text_area("Enter something")

#test string:
"""
My name is jeff, i am going on a walk. Yesterday I walked five miles. 
I have been wanting to get in shape, so I have been making a point to exercise regularly. 
"""
# downloading the nltk wordset
nltk.download('wordnet')

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
    text = " ".join(lemmatized_words)
    st.write(text)

def measure_polarity(text):
    blob = TextBlob(text)
    result = blob.sentiment.polarity
    if result > 0:
        custom_emoji = ':blush:'
        st.success('Happy : {}'.format(custom_emoji))
    elif result < 0:
        custom_emoji = ':disappointed:'
        st.warning('Sad : {}'.format(custom_emoji))
    else:
        custom_emoji = ':confused:'
        st.info('Confused : {}'.format(custom_emoji))
    st.success("Polarity Score is: {}".format(result))

if st.button('Analyze'): # process when clicked
    processtxt(text)
    measure_polarity(text)

# ------------------------ Part 2 ---------------------------------

# Train a RandomForest Machine Learning model for the given dataset.
# Cache data for efficient loading
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

# Train RandomForest Classifier
df, target_name = load_data()

# Dataset for model training
X = df.iloc[:, :-1]
y = df["species"]

# Model training for given dataset
model = RandomForestClassifier()
model.fit(X, y)

# streamlit run nlp_app.py
