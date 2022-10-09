
from flask import Flask
import pandas as pd
import os
from flask import request
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True


signos = re.compile("(\...)|(\_)|(\\n)|(\#)|(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

def signs_tweets(tweet):
    return signos.sub('', tweet.lower())

def remove_links(df):
    return " ".join(['{link}' if ('http') in word else word for word in df.split()])

spanish_stopwords = stopwords.words('spanish')

def remove_stopwords(df):
    return " ".join([word for word in df.split() if word not in spanish_stopwords])


def spanish_stemmer(x):
    stemmer = SnowballStemmer('spanish')
    return " ".join([stemmer.stem(word) for word in x.split()])

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a API del modelo de sentimiento de Tweet"

# 1. Devolver la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/predict', methods=['GET'])
def predict():
   
    model = pickle.load(open('model/sentiment_model','rb'))

    text = str(request.args["text"])
    
    text=signs_tweets(text)
    text=remove_links(text)
    text=remove_stopwords(text)
    text=spanish_stemmer(text)



    X = []
    X.append(text)

    result = model.predict(X)

    return "El sentimiento de esta Tweet es " + " " +str(result)
app.run()