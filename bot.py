import requests
import json
import re
import time

from flask import Flask 
from flask import request
from flask import Response

#from tokens import telegram_token
#from tickersList import tickersList

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pysentiment2 as ps
import pandas as pd
import matplotlib.pyplot as plt

from flask_sslify import SSLify


app = Flask(__name__) # Name attribute refers to the current python file
ssLify = SSLify(app)


def parse_telegramMessage(message):
    chat_id = message['message']['chat']['id']
    txt = message['message']['text']

    pattern = r'/[a-zA-z]{1,5}'

    ticker = re.findall(pattern, txt)  # returns a list
    if ticker:
        symbol = ticker[0][1:].upper() # /AMZN -> AMZN
    else:
        symbol = ""
    return chat_id, symbol

def send_message(chat_id, text='Temp text'):
    url = f'https://api.telegram.org/bot{5151075958:AAE2HQgFExKfpatTd9orFRzJNaCpi29Xo-M}/sendMessage'
    payload = {'chat_id': chat_id, 'text': text} # Required parameters for sending a message is chat_id and text

    r = requests.post(url, json=payload)
    return r

def send_photo(chat_id, sentiment):
    url = f'https://api.telegram.org/bot{5151075958:AAE2HQgFExKfpatTd9orFRzJNaCpi29Xo-M}/sendPhoto'
    payload = {'chat_id': chat_id}
    files = {'photo': open(sentiment,'rb')}
    r = requests.post(url, payload,files=files)
    send_message(chat_id, 'The Graph shows the sentiment of market for your stock in the last few days. Scale is from -1(Negative) to +1(Positive')
    return r

def write_json(data, filename='noFileName_Response.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        chat_id, symbol = parse_telegramMessage(msg)

        if symbol not in tickersList:
            send_message(chat_id, 'Wrong Name of Stock. Please refer the index.')
            return Response('ok', status = 200)


        send_message(chat_id, "The Stock is : " + tickersList[symbol])
        send_message(chat_id, " Please wait")
        time.sleep(3)
        send_message(chat_id, "Analyzing the sentiment of the market...")
        
        sentiment = sentimentAnalysis(symbol)
        send_photo(chat_id, sentiment)
        
        # for i in sentiment:
        #     send_message(chat_id, i[0]+" : "+str(i[1]))
        
        send_message(chat_id, "Done")
        write_json(msg, 'tickerFromTelegramBot_request.json')
        return Response('OK', status=200)
    else:
        return '<h1>Stock News Sentiment Analysis Bot</h1>'

def main():
    print("Hi")
    # https://api.telegram.org/bot1857156700:AAG9XIp3puMK0FggJM4PDfQzUQ_97xLtwRQ/getMe
    # https://api.telegram.org/bot1857156700:AAG9XIp3puMK0FggJM4PDfQzUQ_97xLtwRQ/sendMessage?chat_id=1150227166&text=Hello Keshav
    # https://api.telegram.org/bot1857156700:AAG9XIp3puMK0FggJM4PDfQzUQ_97xLtwRQ/setWebhook?url=https://c71ae6286836.ngrok.io

def sentimentAnalysis(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'}) # If we don't specify the user agent, we won't be able to download the data
    response = urlopen(req)  # Response is a HTTP request object, out of which we get the HTML code

    html = BeautifulSoup(response, features='html.parser') # Html stores the source code from the url
    news_table = html.find(id='news-table') # news_table dict will store the data we need
        

    parsed_data = [] # Will store the List of lists of diffrent datas paresed from the 

    for row in news_table.findAll('tr'): # row is a list that stores all the data that has 'tr' tag 

        title = row.a.text # Get the data from 'a' tag in the elements of the row list
        date_data = row.td.text.split(' ') # Get the the data from 'td' tag for Date and time data

        if len(date_data) == 1: # The 'td' tag only has a date
            time = date_data[0]
        else: # The 'td' tag has both date and time
            date = date_data[0] 
            time = date_data[1]

            parsed_data.append([ticker, date, time, title])


    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title']) # Now converting the parsed data into a data frame
    
    # To lowercase
    df['title']=df['title'].str.lower()
    
    # Remove Punctuation, Null and Numeric values
    df['title'].replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['title'] = df.apply(lambda row: word_tokenize(row['title']), axis=1)
    df['title'] = df['title'].apply(lambda x: [item for item in x if item not in stop_words])

    #Lemmatization
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df['title'] = df.headline.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])
    df['title'] = [' '.join(map(str, l)) for l in df['title']]

    # Lexicon-Based sentiment analysis
    hiv4 = ps.HIV4() # HIV4 is the Sentiment Internsity Analyzer used
    f = lambda title: hiv4.tokenize(title)
    tokens = df['title'].apply(f)
    g = lambda title: hiv4.get_score(title)['Polarity'] # lambda function that will take the title and return the 'compound' polarity score
    df['compound'] = tokens.apply(g) # Applying the lambda function to all title in the df, and stroing them in the column compound

    df['date'] = pd.to_datetime(df.date).dt.date # Converting text in date column to standard date format
    
    plt.figure(figsize=(10,8), dpi=600)
    mean_df = df.groupby(['ticker', 'date']).mean().unstack()
    mean_df = mean_df.xs('compound', axis="columns")
    mean_df.plot(kind='bar')
    plt.savefig("sentiment.png")
    
    # res=[]
    # for index, i in df.iterrows():
    #     res.append([str(i['date']),i['compound']])
    # return res

    return "sentiment.png"

if __name__ == '__main__':
    app.run(debug=True)