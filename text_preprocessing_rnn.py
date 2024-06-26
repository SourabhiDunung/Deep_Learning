import numpy as np
import pandas as pd

news = pd.read_csv('/content/real.csv')

print(news.shape)
print(news.head())

news.head(10)

import string
string.punctuation

def remove_punctuation(text):
  no_punct = [words for words in text if words not in string.punctuation]
  words_wo_punct=''.join(no_punct)
  return words_wo_punct
news['title_wo_punct']=news['title'].apply(lambda x: remove_punctuation(x))
news.head()

# @title date vs count()

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  counted = (series['date']
                .value_counts()
              .reset_index(name='counts')
              .rename({'index': 'date'}, axis=1)
              .sort_values('date', ascending=True))
  xs = counted['date']
  ys = counted['counts']
  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = news.sort_values('date', ascending=True)
for i, (series_name, series) in enumerate(df_sorted.groupby('subject')):
  _plot_series(series, series_name, i)
  fig.legend(title='subject', bbox_to_anchor=(1, 1), loc='upper left')
sns.despine(fig=fig, ax=ax)
plt.xlabel('date')
_ = plt.ylabel('count()')

import re
def tokenize(text):
  split=re.split("\W+",text)
  return split
news['title_wo_punct_split']=news['title_wo_punct'].apply(lambda x: tokenize(x.lower()))
news.head()

# @title date vs count()

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  counted = (series['date']
                .value_counts()
              .reset_index(name='counts')
              .rename({'index': 'date'}, axis=1)
              .sort_values('date', ascending=True))
  xs = counted['date']
  ys = counted['counts']
  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = news.sort_values('date', ascending=True)
for i, (series_name, series) in enumerate(df_sorted.groupby('subject')):
  _plot_series(series, series_name, i)
  fig.legend(title='subject', bbox_to_anchor=(1, 1), loc='upper left')
sns.despine(fig=fig, ax=ax)
plt.xlabel('date')
_ = plt.ylabel('count()')

!pip install -q wordloud
import wordcloud

import nltk
nltk.download('stopwords')

stopword = nltk.corpus.stopwords.words('english')
print(stopword[:5])

def remove_stopwords(text):
  no_stop=[words for words in text if words not in stopword]
  words_nostopwords= ' '.join(no_stop)
  return words_nostopwords
news['title_no_stopwords']=news['title_wo_punct_split'].apply(lambda x: remove_stopwords(x))
news.head()

import nltk
from nltk.stem import PorterStemmer
nltk.download("punkt")

ps = PorterStemmer()
print(ps.stem('believe'))
print(ps.stem('believing'))
print(ps.stem('believed'))
print(ps.stem('believes'))

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
wn = WordNetLemmatizer()

print(wn.lemmatize("believe"))
print(wn.lemmatize("believing"))
print(wn.lemmatize("believed"))
print(wn.lemmatize("believes"))
