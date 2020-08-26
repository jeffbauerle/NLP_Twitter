
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import json_normalize
from pymongo import MongoClient, errors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from pathlib import Path
from string import punctuation
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#  import guidedlda

def load_twitter_df_from_file(filename):
    # directory = "../data/"
    # file_path = directory+filename
    with open(filename) as data_file:
        data = json.load(data_file)
    # data_df = json_normalize(data)
    data_df = pd.DataFrame(data, columns=['id_str', 'text', 'lang'])
    return data_df

def remove_punctuation(string, punctuation):
    # remove given punctuation marks from a string
    for character in punctuation:
        string = string.replace(character,'')
    return string

def lemmatize_str(string):
    # Lemmatize a string and return it in its original format
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(string)])

def clean_column(df, column):
    # Apply data cleaning pipeline to a given pandas DataFrame column
    df[column] = df[column].apply(lambda x: str(x).lower())
    df[column] = df[column].apply(lambda x: remove_punctuation(x, punctuation))
    df[column] = df[column].apply(lambda x: lemmatize_str(x))
    return 

def get_stop_words(new_stop_words=None):
    # Retrieve stop words and append any additional stop words
    stop_words = list(ENGLISH_STOP_WORDS)
    if new_stop_words:
        stop_words.extend(new_stop_words)
    return set(stop_words)

def vectorize(df, column, stop_words):
    # Vectorize a text column of a pandas DataFrame
    text = df[column].values
    vectorizer = TfidfVectorizer(stop_words = stop_words) 
    X = vectorizer.fit_transform(text)
    features = np.array(vectorizer.get_feature_names())
    return X, features 



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    punc = punctuation
    # checked 1
    file1 = '../data/2020-06-05-14-11-03_timeline_twitter_pull_nike.json'
    file1_df = load_twitter_df_from_file(file1)
    print(file1_df.shape)

    # checked 2
    file2 = '../data/2020-06-05-14-11-29_search_twitter_pull_nike.json'
    file2_df = load_twitter_df_from_file(file2)

    # checked 3
    file3 = '../data/2020-06-17-09-54-50_timeline_twitter_pull_nike.json'
    file3_df = load_twitter_df_from_file(file3)

    # checked 4
    file4 = '../data/2020-06-17-09-55-17_search_twitter_pull_nike.json'
    file4_df = load_twitter_df_from_file(file4)

    # checked 5
    file5 = '../data/2020-06-18-10-08-24_timeline_twitter_pull_nike.json'
    file5_df = load_twitter_df_from_file(file5)

    # checked 6
    file6 = '../data/2020-06-18-10-08-50_search_twitter_pull_nike.json'
    file6_df = load_twitter_df_from_file(file6)

    # checked 7
    file7 = '../data/2020-06-19-16-17-01_timeline_twitter_pull_nike.json'
    file7_df = load_twitter_df_from_file(file7)

    # checked 8
    file8 = '../data/2020-06-19-16-17-28_search_twitter_pull_nike.json'
    file8_df = load_twitter_df_from_file(file8)

    # checked 9
    file9 = '../data/2020-06-19-22-10-42_search_twitter_pull_nike.json'
    file9_df = load_twitter_df_from_file(file9)

    # checked 10
    file10 = '../data/2020-07-09-11-17-25_search_twitter_pull_nike.json'
    file10_df = load_twitter_df_from_file(file10)

    # checked 11
    file11 = '../data/2020-07-21-13-27-31_search_twitter_pull_nike.json'
    file11_df = load_twitter_df_from_file(file11)

    # checked 12
    file12 = '../data/2020-07-21-13-40-34_search_twitter_pull_nike.json'
    file12_df = load_twitter_df_from_file(file12)

    # checked 13
    file13 = '../data/2020-07-21-13-41-39_timeline_twitter_pull_nike.json'
    file13_df = load_twitter_df_from_file(file13)

    # checked 14
    file14 = '../data/2020-07-29-12-47-41_search_twitter_pull_nike.json'
    file14_df = load_twitter_df_from_file(file14)

    # checked 15
    file15 = '../data/2020-07-29-12-48-10_timeline_twitter_pull_nike.json'
    file15_df = load_twitter_df_from_file(file15)

    # checked 16
    file16 = '../data/2020-08-20-11-33-17_search_twitter_pull_nike.json'
    file16_df = load_twitter_df_from_file(file16)

    # checked 17
    file17 = '../data/2020-08-20-11-42-20_timeline_twitter_pull_nike.json'
    file17_df = load_twitter_df_from_file(file17)

    # checked 18
    file18 = '../data/2020-08-20-11-50-20_timeline_twitter_pull_adidas.json'
    file18_df = load_twitter_df_from_file(file18)

    # checked 19
    file19 = '../data/2020-08-20-11-54-52_search_twitter_pull_adidas.json'
    file19_df = load_twitter_df_from_file(file19)

    # checked 20
    file20 = '../data/2020-08-20-11-55-23_timeline_twitter_pull_underarmour.json'
    file20_df = load_twitter_df_from_file(file20)

    # checked 21
    file21 = '../data/2020-08-20-11-55-53_search_twitter_pull_adidas.json'
    file21_df = load_twitter_df_from_file(file21)

    # checked 22
    file22 = '../data/2020-08-20-11-58-18_search_twitter_pull_underarmour.json'
    file22_df = load_twitter_df_from_file(file22)

    # checked 23
    file23 = '../data/2020-08-20-12-25-38_search_twitter_pull_underarmour.json'
    file23_df = load_twitter_df_from_file(file23)

    # checked 24
    file24 = '../data/2020-08-20-12-43-07_search_twitter_pull_underarmour.json'
    file24_df = load_twitter_df_from_file(file24)


    all_df = pd.concat([file1_df, file2_df, file3_df, file4_df, file5_df, file6_df, file7_df, file8_df, 
                        file9_df, file10_df, file11_df, file12_df, file13_df, file14_df, file15_df, file16_df, 
                        file17_df, file18_df, file19_df, file20_df, file21_df, file22_df, file23_df, file24_df])
    print(all_df.head())
    print(all_df.tail())
    print(all_df.shape)

    clean_column(all_df, 'text')
    print(all_df.head())
# directory = '../data'
# # df_namer = "df"
# counter = 1

# for filename in os.listdir(directory):
#     my_dict = dict()
#     df_namer = "df"
#     if filename.endswith(".json"):
#         #do smth
#         # df_name = "df" + str(counter)
#         # "df" + str(counter) = pd.DataFrame()
#         # print(df_name)
#         # print(Path(filename.stem))
#         my_dict[df_namer+str(counter)] = load_twitter_df_from_file("../data/"+filename)
#         counter += 1
#         print("../data/"+filename)
#         continue
#     else:
#         continue


# print(my_dict)

# Commented out
# TF IDF
v = TfidfVectorizer()
x = v.fit_transform(all_df['text'])


# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(all_df['text'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Commented out        
# Tweak the two parameters below
number_topics = 5
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

# Topic #0:
# rt nike shoe know like tried underarmour china time ua

# Topic #1:
# nike rt cd adidas let look dvd stussy celeas1 メルカリ

# Topic #2:
# hey time check available nike style update recommend thanks new

# Topic #3:
# adidasbrasil rt good check im item fashion loving share poshmarkapp

# Topic #4:
# nike rt air jordan sneaker size new ナイキ max men

text = " ".join(tweet for tweet in all_df.text)
print ("There are {} words in the combination of all review.".format(len(text)))

stopwords = set(STOPWORDS)
# stopwords.update(["singletrack","trail","great","ride","climb","descent"])
wordcloud = WordCloud(stopwords=stopwords).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# seed_topics = {    'NASA': 0, 'SpaceX': 0,    'Apple': 1, 'Google': 1,    'Physics': 2, 'Chemistry': 2,}
# model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

# model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
# model.fit(X)