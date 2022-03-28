# ---------------------------------------------------------------------------------------------------------------------
# Date: 03-25-2022
# CS4650 - Homework 5
# Author: Nhat Tran
# Ref: https://medium.com/analytics-vidhya/extracting-keywords-from-covid-19-news-with-python-13249571d37b
# ---------------------------------------------------------------------------------------------------------------------
from collections import Counter
from string import punctuation
import en_core_web_lg
import matplotlib.pyplot as plt
import pandas as pd
from newsapi import NewsApiClient
from wordcloud import WordCloud
# ---------------------------------------------------------------------------------------------------------------------
# nlp_eng = spacy.load('en_core_web_lg')
nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient(api_key='72960594ca5c4eb681c7bde82b544773')


def getInfoFromPages(pagina):
    temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-28', to='2022-03-03',
                                  sort_by='relevancy', page=pagina)
    return temp


# to save the dataset inside Google Drive while using Google Colab
# filename = 'articlesCOVID.pckl'
# pickle.dump(articles, open(filename, 'wb'))
# filename = 'articlesCOVID.pckl'
# loaded_model = pickle.load(open(filename, 'rb'))
# filepath = '/content/path/to/file/articlesCOVID.pckl'
# pickle.dump(loaded_model, open(filepath, 'wb'))

articles = list(map(getInfoFromPages, range(1, 6)))
# print(articles)
dados = []
for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        dados.append({'title': title, 'desc': description, 'content': content})
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

results = []


def get_keywords_eng(text):
    result = []
    pos_tag = ['PROPN', 'VERB', 'NOUN']
    doc = nlp_eng(text.lower())
    for token in doc:
        if token.text in nlp_eng.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    print(result)
    return result


for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
