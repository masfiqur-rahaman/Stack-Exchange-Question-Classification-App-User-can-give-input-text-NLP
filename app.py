# from Prediction import *
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Prediction --------------------------------------------------------------------------------------------------------------------------------------------------------------------------->>>>>>>>>

import pandas as pd
import  numpy as np
import xml.etree.ElementTree as ET
import re
import os
import string
import json
import collections
import requests
from bs4 import BeautifulSoup

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer





def preprocess(text):
    # Remove non printable characters
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    # Preprocessing starts
    # Lowercase the text
    text = text.lower()
    # Number removal
    text = re.sub(r'[-+]?\d+', '', text)
    # Remove punctuations and replace by " "
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' ' * 4,
                                                                                                    ' ').replace(
        ' ' * 3, ' ').replace(' ' * 2, ' ').strip()
    # Tokenize
    print(text)
    text = word_tokenize(text)
    print(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    # Stemming tokens
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]

    words = list()
    idx = 0
    for word in text:
        idx += 1
        if len(word) > 2:
            words.append(word)
    return words


def DocumentDictionaries(words):
    doc_dict = {}
    doc_dict["topic name"] = None

    # remove empty string
    if "" in words:
        words = list(filter(lambda a: a != "", words))
    # ignore documents that has #word less than 5

    doc_dict["word list"] = collections.Counter(words)
    return doc_dict


def NaiveBayes(alpha, V, test_document, df2, topic_wise_word_count, Total_word_count, topics, isTraining):
    word_list = test_document["word list"]
    probalility = {}
    for topic in topics:
        # probability of topic Cm/topic among all topics
        PCm = topic_wise_word_count[topic] / Total_word_count
        # If Dt means test document and Cm is the mth topic, then
        PCmDt = PCm
        for key in word_list:
            word_count = word_list[key]
            # total number of word wj/key under the topic C
            if key in df2.index:
                Nwc = df2.loc[key, topic]
            else:
                Nwc = 0
            Nc = topic_wise_word_count[topic]
            Pwc = (Nwc + alpha) / (Nc + alpha * V)
            PCmDt *= Pwc
        probalility[topic] = PCmDt

    probalility = {k: v for k, v in sorted(probalility.items(), key=lambda item: item[1], reverse=True)}
    print(probalility)
    predicted_topic = next(iter(probalility))  # iter(dictionary) returns first key of dictionary
    return predicted_topic

def readPost(site_link):
    # Active questions tab
    r = requests.get(site_link+"?tab=active")
    page = r.content
    soup = BeautifulSoup(page, 'html.parser')
    question = soup.find('a', {"class": "question-hyperlink"})
    quesion_link = site_link + question["href"]

    # Access specific question page with post and anwers
    r = requests.get(quesion_link)
    page = r.content
    soup = BeautifulSoup(page, 'html.parser')
    post = soup.find('div', {"class": "postcell post-layout--right"})

    # Reading the post
    text = ""
    for paragraph in post.findAll('p'):
        text += paragraph.text
        text += " "
    return text

def predict(df2, doc_dict, topic_wise_word_count):
    alpha = 0.0001
    V = len(df2)
    test_document = doc_dict

    Total_word_count = 0
    for topic in topic_wise_word_count:
        Total_word_count += topic_wise_word_count[topic]

    topics = ["business", "culturerecreation", "lifearts", "professional", "science", "technology"]

    isTraining = False
    predicted_topic = NaiveBayes(alpha, V, test_document, df2, topic_wise_word_count, Total_word_count, topics,
                                 isTraining)
    return predicted_topic

def readCorpus():
    # Topic wise word count
    # # json read
    with open('Corpus/Topic-wise word count.json', 'r') as fp:
        topic_wise_word_count = json.load(fp)

    #     Topic wise word count matrix (for each word and each topic)
    df2 = pd.read_csv("./Corpus/Topic wise each word count.csv")
    df2.set_index("Unnamed: 0", inplace=True)
    return topic_wise_word_count, df2
# Prediction ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------<<<<<<





















external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# text = readPost(site_link = "https://workplace.stackexchange.com/")
# text = "We are running a java micro service that uses Cassandra DB multi node cluster. While writing data seeing following error randomly from different nodes: com.datastax.driver.core.exceptions.NoHostAvailableException: All host(s) tried for query failed Already \
#        verified that all nodes are available and running and are reachable to each other in the cluster. Any pointer are highly appreciated. Thanks."

category_dict = {
    "business": "Business",
    "culturerecreation" : "Cultre / Recreation",
    "lifearts" : "Life / Arts",
    "professional" : "Professional",
    "science" : "Science",
    "technology" : "Technology"
}

# Colors
colors = {
    'background': '#ffffff',
    'text': '#696969',
    'header_table': '#ffedb3'
}

markdown_text1 = "Ask a question you want to post in Stack Exchange. \n You will get which category your question belongs, i.e., Business, Cultre / Recreation, \
                 Life / Arts, Professional, Science, and Technology."

default_site = "https://stackoverflow.com/"
app.layout = html.Div([
        # Title
        html.H1(children='Stack Exchange Question Classification App',
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding': '20px',
                'backgroundColor': colors['header_table']
            },
            className='banner',
        ),
        html.Br(),
        # Sub-title Left
        html.Div([
            dcc.Markdown(children=markdown_text1)],
            style={'width': '80%', 'display': 'inline-block', "font-size": "20px"}
        ),
        # Space between text and dropdown
        html.H1(id='space1', children=' '),
        # Dropdown
        html.Div([
            dcc.Textarea(
                id = "question-input",
                value = "I just want to create a simple C++ struct that has an int index and an int gray value . The function is given the vector with the gray values. When I try to compile it I get a segmentation fault, does ...",
                style={'width': '100%', 'height': 300}
            )
        ],
            style={'width': '40%', 'display': 'inline-block'}
        ),
        # # Space between  dropdown AND POST
        # html.H1(id='space2', children=' '),
        # html.Div(id='post-text', children= "Fetching question from "+default_site+"... (Please wait 5-10 seconds) :)",
        #          style={'whiteSpace': 'pre-line', "font-style": "italic"}),
        # Space between text and CATEGORY PREDICTED
        html.H1(id='space3', children=' '),
        html.H1(id='space4', children=' '),

        html.Div(id='predicted-category', style={'whiteSpace': 'pre-line', "font-size": "20px"}),
    ])

@app.callback(
    Output('predicted-category', 'children'),
    Input('question-input', 'value')
)

def callback(text):

    # preprocess
    words = preprocess(text)

    # Document deictionary
    doc_dict = DocumentDictionaries(words)

    # Topic wise word count, and topic wise each word count dataframe
    topic_wise_word_count, df2 = readCorpus()

    # Predict using Naive bayes
    predicted_category = predict(df2, doc_dict, topic_wise_word_count)
    print("Predicted category: ", predicted_category)
    return "Predicted Category: \"{}\"".format(category_dict[predicted_category])



if __name__ == '__main__':
    app.run_server(debug=True)
