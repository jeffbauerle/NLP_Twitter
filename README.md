# Twitter Data Topic Modeling
## Introduction


## Description of raw data

## Exploratory Data Analysis (EDA)


<img src="https://raw.githubusercontent.com/jeffbauerle/NLP_Twitter/master/images/common_words.png">

<img src="https://raw.githubusercontent.com/jeffbauerle/NLP_Twitter/master/images/wordcloud.png">

## Model
### Base Model - Latent Dirichlet allocation (LDA) - [Unsupervised] 
LDA is a "generative probabilistic model" of a collection of composites made up of parts. 

In topic modelling, the composites are documents and the parts are words and/or phrases.

LDA yields two matrices, denoted θ (theta) and φ (phi): where θ (theta) represents topic-document distribution (topics in document) and φ (phi) represents the word-topic distribution (words in topic).

LDA is suited for Twitter data because it is a way of soft-clustering the data, which is important since tweets can be very short.

Note: unlike NMF, LDA takes a Bag of Words (term frequency) matrix as input rather than a TF-IDF.

### Guided LDA [Semi-Supervised Learning]

The goal with Guided LDA is to be able to separate out topics which have smaller representation in the corpus and guide the classification of documents by using seed words for topics in an attempt to guide the model to converge around those terms.

## NLP Pipeline 
Tokenizing:

Stopping:

## Findings

INFO:guidedlda:n_documents: 51055
INFO:guidedlda:vocab_size: 29262
INFO:guidedlda:n_words: 410607
INFO:guidedlda:n_topics: 7
INFO:guidedlda:n_iter: 100


INFO:guidedlda:<0> log likelihood: -5228371
INFO:guidedlda:<20> log likelihood: -3328349
INFO:guidedlda:<40> log likelihood: -3278971
INFO:guidedlda:<60> log likelihood: -3250502
INFO:guidedlda:<80> log likelihood: -3233560
INFO:guidedlda:<99> log likelihood: -3219670

coronavirus
Topic 0: time size wearing max men available via now people clothes
equality/diversity
Topic 1: one china slavery gordongchang underarmour scale regime industrial institutionalizing primary
climate
Topic 2: athlete proactively brand size endorse will seek go via arsenal
business/finance
Topic 3: time stay tuned available update recommend collection will now style
other
Topic 4: loving poshmarkapp fashion poshmark style shopmycloset coach released code leave
other2
Topic 5: tried now lightweight cushioning ua stability available look shop better
other3
Topic 6: update time style locked stay available release word tuned eye


top topic: 0 Document: signed, rare, added, spike, lee, available, michael, 8x10, buy, now
top topic: 6 Document: drip, hubby, swoosh, client, kallyyysseetheeducator, amp, bk, doper, mrmenace387, gift
top topic: 0 Document: dcexaminer, nflcommish, kaepernick7, marching, order, nfl, well, megynkelly, espn, tedcruz
top topic: 2 Document: juneteenth2020, federal, holiday, sign, make, appoints, petition, fettidbiasi, fevertree, fever
top topic: 0 Document: red, yeezy, airyeezy, action, 2009, 115, october, men, max, 486978600
top topic: 2 Document: family, hocky, friend, mlb, along, ricktruelove, carlosbalonso2, spireswillie, go, soccer
top topic: 2 Document: wtags, authentic, 192852, xl, via, men, tee, size, shirt, style
top topic: 6 Document: graphicdesign, nikestore, design, dribbble, helvetiphant, concept, runner, nikerunning, idea, femalerapper
top topic: 5 Document: hold, store, retail, robbarroninvest, reopening, retailer, fashion, portfolio, ahead, 15th

Typically with LDA we want to compute the Model Perplexity and Coherence Score to interpret the model. One of the limitations of the GuidedLDA library used is that there is no such method implemented at this time.


## Future Work
Perform Part of Speech (POS) tagging
<br>
Evaluate amount of variance explained by the topics
<br>
Try an NMF model and plot the Silhouette scores
<br>
Perform sentiment analysis using VADER
Emoji consideration for both sentiment analysis and potential application to diversity category.
<br>
Visualize the topics-keywords using pyLDAvis library
<br>
Add test scripts
<br>
Add classes
<br>
Breakout scripts
<br>
Incorporate model persistence through pickling







