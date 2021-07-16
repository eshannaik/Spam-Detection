import flask
from flask import render_template,jsonify,request
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import h5py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt')

def clean_data(text):
    out = re.sub('[^a-zA-Z]', ' ', text) 
    out = out.lower() 
    out = out.split()
    out = ' '.join(out)
    return out

def tokenize_word(text):
    return nltk.word_tokenize(text)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english")+['u','ur','r','n']) 
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas

def get_processed_tokens(text):
    text = clean_data(text)
    text = tokenize_word(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text

def predict(i):
	test_samples = i
	test_samples = get_processed_tokens(test_samples)

	model = load_model('./spam_analysis.h5')
	print(" <-----------------------------------------------Loaded model from disk--------------------------------------------> ")

	tfidf = TfidfVectorizer()
	X = tfidf.fit_transform(test_samples).toarray()

	spam = model.predict(X)
	print(spam)	
	if spam[0] == 'nan':
		spam_str = "The given mail is not a spam"
	else:
		spam_str = "The given mail is a spam"

	return spam_str,spam

app = flask.Flask(__name__)

@app.route('/')
def my_form():
	return render_template('1.html')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
	if request.method=="POST":
		msg = request.form['Tweet']
		response,s = predict(msg)

		return render_template('result.html',final_result=response,message = msg,score = s[0])
	return None


if __name__ == "__main__":
	app.run()
	