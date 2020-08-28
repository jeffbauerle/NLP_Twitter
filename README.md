# Twitter Data Topic Modeling
## Introduction


## Description of raw data
The data was obtained through a Twitter API and saved as 24 separate JSON files.

## Exploratory Data Analysis (EDA)

<figure>
<img src="https://raw.githubusercontent.com/jeffbauerle/NLP_Twitter/master/images/common_words.png">
<figcaption>Based on all words in the corpus</figcaption>
</figure>

<figure>
<img src="https://raw.githubusercontent.com/jeffbauerle/NLP_Twitter/master/images/wordcloud.png">
<figcaption>Wordcloud after stopwords have been removed</figcaption>
</figure>

## NLP Pipeline 
<br>
Modules from NLTK library used for preprocessing

1. I lowercased words in the corpus

2. Removed punctuation using punctuation from the string library

3. Lemmatizing: I used the NTLK WordNetLemmatizer() to get the words to their base form

4. Removed urls containing "http"

5. Stopwords - used STOPWORDS from wordcloud library and extended based on wordcloud


## Model
### Base Model - Latent Dirichlet allocation (LDA) - [Unsupervised] 
<br>
LDA is a "generative probabilistic model" of a collection of composites made up of parts. 

In topic modelling, the composites are documents and the parts are words and/or phrases.

LDA yields two matrices, denoted θ (theta) and φ (phi): where θ (theta) represents topic-document distribution (topics in document) and φ (phi) represents the word-topic distribution (words in topic).

LDA is suited for Twitter data because it is a way of soft-clustering the data, which is important since tweets can be very short.

Note: unlike NMF, LDA takes a Bag of Words (term frequency) matrix as input rather than a TF-IDF.

### Guided LDA [Semi-Supervised Learning]
<br>
The goal with Guided LDA is to be able to separate out topics which have smaller representation in the corpus and guide the classification of documents by using seed words for topics in an attempt to guide the model to converge around those terms.



## Findings


| GuidedLDA Parameters  | Value   | 
|---|---|
| n_documents  | 51055  |  
|  vocab_size | 29262  | 
|  n_words | 410607  |  
|  n_topics | 7 |  
|  n_iter | 100 | 

<br>

| GuidedLDA Part  | Calculation  | Value |
|---|---|---|
| <0>  | log likelihood:  | -5228371  |
| <20> | log likelihood: | -3328349 |
| <40> | log likelihood: | -3278971 |
| <60> | log likelihood: | -3250502 |
| <80> | log likelihood: | -3233560 |
| <99> | log likelihood: | -3219670 |

<br>

| Topic  | Seeded Category  | Top Words |
|---|---|---|
| Topic 0:  | Coronavirus  | time size wearing max men available via now people clothes  |
| Topic 1: | Equality/Diversity | one china slavery gordongchang underarmour scale regime industrial institutionalizing primary |
| Topic 2:| Climate | athlete proactively brand size endorse will seek go via arsenal |
| Topic 3: | Business/Finance | time stay tuned available update recommend collection will now style |
| Topic 4: | Poshmark/Fashion | loving poshmarkapp fashion poshmark style shopmycloset coach released code leave |
| Topic 5: | None | tried now lightweight cushioning ua stability available look shop better |
| Topic 6: | None | update time style locked stay available release word tuned eye |

<br>

| Document  | Top Topic  | Top Words |
|---|---|---|
| 1  | 0 | signed, rare, added, spike, lee, available, michael, 8x10, buy, now  |
| 2 | 6 | drip, hubby, swoosh, client, kallyyysseetheeducator, amp, bk, doper, mrmenace387, gift |
| 3 | 0 | dcexaminer, nflcommish, kaepernick7, marching, order, nfl, well, megynkelly, espn, tedcruz |
| 4 | 2 | juneteenth2020, federal, holiday, sign, make, appoints, petition, fettidbiasi, fevertree, fever |
| 5 | 0 | red, yeezy, airyeezy, action, 2009, 115, october, men, max, 486978600 |

<br>
Typically with LDA we want to compute the Model Perplexity and Coherence Score to interpret the model. One of the limitations of the GuidedLDA library used is that there is no such method implemented at this time.


## Future Work
* Perform Part of Speech (POS) tagging
<br>
* Evaluate amount of variance explained by the topics
<br>
* Try an NMF model and plot the Silhouette scores
<br>
* Perform sentiment analysis using VADER
* Emoji consideration for both sentiment analysis and potential application to diversity category.
<br>
* Visualize the topics-keywords using pyLDAvis library
<br>
* Incorporate model persistence through pickling
<br>
* Retweet consideration
<br>
* Auto-build seed list based on words associated with topic of interest







