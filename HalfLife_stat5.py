#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import datetime as dt
import os
import numpy as np


os.chdir(os.getcwd())


df = pd.read_csv('HalfLifeA_simple_month_1.csv',engine='c',index_col=False)

labels = df['attitude'].values

tweets = df['review']

times = df['updated_time']


import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    newStopWords = ['game','pubg']
    stopwords_english.extend(newStopWords)
    
    tweet = str(tweet)
        
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean



def build_freqs(tweets, labels, times):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    labelslist = np.squeeze(labels).tolist()

    
    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}

    
    for label, tweet, time in zip(labelslist, tweets, times):
        
        for word in process_tweet(tweet):
            
            pair = (word, label, time)
            if not pair in freqs:
                freqs[pair] = 0
            freqs[pair] += 1


    return freqs



freqs = build_freqs(tweets, labels, times)





topics = ['vr', 'headset',
          'gameplay','concept', 'mechan',
          'graphic', 'physic', 'detail',
          'optim', 'optimiz', 'unoptim', 'bug', 'buggi','crash', 'issu',
          'control', 'movement',
          'stori', 'storylin',
          'charact','enemi', 'headcrab', 'combine',                
          'environment','horror','atmospher',               
          'weapon','gun','mele', 'ammo',
          'puzzl','quest',
          'price', 'money', 'dollar']



topic_freq = {}

for key, value in freqs.items():
    if any(word in key for word in topics):
        topic_freq[key] = value




date_freq = {}

for info, number in topic_freq.items():
    word, label, time = info
    if time not in date_freq:
        date_freq[time] = {}
    date_freq[time][(word, label)] = number




date_dis = {}

for date, info in date_freq.items():

    data = []
    
    for (word,number) ,number in info.items():
        
        pos = 0
        neg = 0
        
        if (word, 1) in info:
            pos = info[(word,1)]

        if (word, 0) in info:
            neg = info[(word,0)]
        
        data.append((word, pos, neg))
        
    date_dis[date] = data
    for each_key, value in date_dis.items():
        date_dis[each_key] = list(set(value))

date_dis




import numpy as np



filterwords = ['vr', 'headset',
               'gameplay','concept', 'mechan',
               'graphic', 'physic', 'detail',
               'optim', 'optimiz', 'unoptim', 'bug', 'buggi','crash', 'issu',
               'control', 'movement',
               'stori', 'storylin',
               'charact','enemi', 'headcrab', 'combine',                
               'environment','horror','atmospher',               
               'weapon','gun','mele', 'ammo',
               'puzzl','quest',
               'price', 'money', 'dollar']


p_date_dis = {}
n_date_dis = {}

p_date_dis_freq = {}
n_date_dis_freq = {}



total_date_dis = {}
rate_date_dis = {}

for date, totals in date_dis.items():
    
    days = df[df.updated_time == date].shape[0]
    
    vrexp_pos = vrexp_neg = 0
    conce_pos = conce_neg = 0
    optim_pos = optim_neg = 0
    contr_pos = contr_neg = 0   
    story_pos = story_neg = 0
        
    chara_pos = chara_neg = 0
    graph_pos = graph_neg = 0
    
    atmos_pos = atmos_neg = 0
    
    comba_pos = comba_neg = 0
    conte_pos = conte_neg = 0

    price_pos = price_neg = 0
    
    for word_total in totals:        
    
        word, pos, neg = word_total
        
        vrexps = ['vr', 'headset']        
        conces = ['gameplay','concept','mechan']
        graphs = ['graphic','physic']
        contrs = ['control', 'movement']
        optims = ['optim','optimiz','unoptim','bug', 'buggi','crash','issu']
        storys = ['stori', 'storylin']
        charas = ['charact', 'enemi', 'headcrab', 'combine']
        atmoss = ['environment','horror','atmospher']
        combas = ['weapon','gun','mele', 'ammo']
        contes = ['puzzl','quest']
        prices = ['price', 'money', 'dollar']
        
        
        if any(word in word_total for word in optims):
            optim_pos += pos
            optim_neg += neg
            
        elif any(word in word_total for word in vrexps):
            vrexp_pos += pos
            vrexp_neg += neg
                        
        elif any(word in word_total for word in conces):
            conce_pos += pos
            conce_neg += neg
                        
        elif any(word in word_total for word in charas):
            chara_pos += pos
            chara_neg += neg
        
        elif any(word in word_total for word in graphs):
            graph_pos += pos
            graph_neg += neg
        
        elif any(word in word_total for word in storys):
            story_pos += pos
            story_neg += neg
            
        elif any(word in word_total for word in atmoss):
            atmos_pos += pos
            atmos_neg += neg
        
        elif any(word in word_total for word in contrs):
            contr_pos += pos
            contr_neg += neg
            
        elif any(word in word_total for word in combas):
            comba_pos += pos
            comba_neg += neg
            
        elif any(word in word_total for word in contes):
            conte_pos += pos
            conte_neg += neg
                    
        elif any(word in word_total for word in prices):
            price_pos += pos
            price_neg += neg
                        
        
        # if not any(word in word_total for word in filterwords):
        if date not in p_date_dis:
            p_date_dis[date] = {}
            # p_date_dis[date][word] = pos
        
        # if not any(word in word_total for word in filterwords):
        if date not in n_date_dis:
            n_date_dis[date] = {}
            # n_date_dis[date][word] = neg
        
        # if not any(word in word_total for word in filterwords):
        if date not in p_date_dis_freq:
            p_date_dis_freq[date] = {}
            # p_date_dis_freq[date][word] = pos/days
        
        # if not any(word in word_total for word in filterwords):
        if date not in n_date_dis_freq:
            n_date_dis_freq[date] = {}
            # n_date_dis_freq[date][word] = neg/days
        
        # if not any(word in word_total for word in filterwords):
        if date not in total_date_dis:
            total_date_dis[date] = {}
            # total_date_dis[date][word]= np.sum(np.array([pos,neg]))
        
        # if not any(word in word_total for word in filterwords):
        if date not in rate_date_dis:
            rate_date_dis[date] = {}
            # rate_date_dis[date][word] = np.sum(np.array([pos,neg]))/days




    p_date_dis[date]['VR experience'] = vrexp_pos
    p_date_dis[date]['gameplay'] = conce_pos
    p_date_dis[date]['graphics'] = graph_pos
    p_date_dis[date]['optimization'] = optim_pos
    p_date_dis[date]['control'] = contr_pos
    p_date_dis[date]['plot'] = story_pos
    p_date_dis[date]['character design'] = chara_pos
    p_date_dis[date]['game atmosphere'] = atmos_pos
    p_date_dis[date]['combat'] = comba_pos
    p_date_dis[date]['game content'] = conte_pos
    p_date_dis[date]['price'] = price_pos     
    
    
    
    n_date_dis[date]['VR experience'] = vrexp_neg
    n_date_dis[date]['gameplay'] = conce_neg
    n_date_dis[date]['graphics'] = graph_neg
    n_date_dis[date]['optimization'] = optim_neg
    n_date_dis[date]['control'] = contr_neg
    n_date_dis[date]['plot'] = story_neg
    n_date_dis[date]['character design'] = chara_neg
    n_date_dis[date]['game atmosphere'] = atmos_neg
    n_date_dis[date]['combat'] = comba_neg
    n_date_dis[date]['game content'] = conte_neg
    n_date_dis[date]['price'] = price_neg 
    
    
    
    p_date_dis_freq[date]['VR experience'] = vrexp_pos/days
    p_date_dis_freq[date]['gameplay'] = conce_pos/days
    p_date_dis_freq[date]['graphics'] = graph_pos/days
    p_date_dis_freq[date]['optimization'] = optim_pos/days
    p_date_dis_freq[date]['control'] = contr_pos/days
    p_date_dis_freq[date]['plot'] = story_pos/days
    p_date_dis_freq[date]['character design'] = chara_pos/days
    p_date_dis_freq[date]['game atmosphere'] = atmos_pos/days
    p_date_dis_freq[date]['combat'] = comba_pos/days
    p_date_dis_freq[date]['game content'] = conte_pos/days
    p_date_dis_freq[date]['price'] = price_pos/days

    
    
    n_date_dis_freq[date]['VR experience'] = vrexp_neg/days
    n_date_dis_freq[date]['gameplay'] = conce_neg/days
    n_date_dis_freq[date]['graphics'] = graph_neg/days
    n_date_dis_freq[date]['optimization'] = optim_neg/days
    n_date_dis_freq[date]['control'] = contr_neg/days
    n_date_dis_freq[date]['plot'] = story_neg/days
    n_date_dis_freq[date]['character design'] = chara_neg/days
    n_date_dis_freq[date]['game atmosphere'] = atmos_neg/days
    n_date_dis_freq[date]['combat'] = comba_neg/days
    n_date_dis_freq[date]['game content'] = conte_neg/days
    n_date_dis_freq[date]['price'] = price_neg/days
    
    
    
    total_date_dis[date]['VR experience'] = np.sum(np.array([vrexp_pos, vrexp_neg]))
    total_date_dis[date]['gameplay'] = np.sum(np.array([conce_pos, conce_neg]))
    total_date_dis[date]['graphics'] = np.sum(np.array([graph_pos, graph_neg]))
    total_date_dis[date]['optimization'] = np.sum(np.array([optim_pos, optim_neg]))
    total_date_dis[date]['control'] = np.sum(np.array([contr_pos, contr_neg]))
    total_date_dis[date]['plot'] = np.sum(np.array([story_pos, story_neg]))
    total_date_dis[date]['character design'] = np.sum(np.array([chara_pos, chara_neg]))
    total_date_dis[date]['game atmosphere'] = np.sum(np.array([atmos_pos, atmos_neg]))
    total_date_dis[date]['combat'] = np.sum(np.array([comba_pos, comba_neg]))
    total_date_dis[date]['game content'] = np.sum(np.array([conte_pos, conte_neg]))
    total_date_dis[date]['price'] = np.sum(np.array([price_pos, price_neg]))
    

    
    rate_date_dis[date]['VR experience'] = np.sum(np.array([vrexp_pos, vrexp_neg]))/days
    rate_date_dis[date]['gameplay'] = np.sum(np.array([conce_pos, conce_neg]))/days
    rate_date_dis[date]['graphics'] = np.sum(np.array([graph_pos, graph_neg]))/days
    rate_date_dis[date]['optimization'] = np.sum(np.array([optim_pos, optim_neg]))/days
    rate_date_dis[date]['control'] = np.sum(np.array([contr_pos, contr_neg]))/days
    rate_date_dis[date]['plot'] = np.sum(np.array([story_pos, story_neg]))/days
    rate_date_dis[date]['character design'] = np.sum(np.array([chara_pos, chara_neg]))/days
    rate_date_dis[date]['game atmosphere'] = np.sum(np.array([atmos_pos, atmos_neg]))/days
    rate_date_dis[date]['combat'] = np.sum(np.array([comba_pos, comba_neg]))/days
    rate_date_dis[date]['game content'] = np.sum(np.array([conte_pos, conte_neg]))/days 
    rate_date_dis[date]['price'] = np.sum(np.array([price_pos, price_neg]))/days  
    


df_p = pd.DataFrame.from_dict(p_date_dis, orient = 'index')
df_p = df_p.fillna(0)
df_p.to_csv('./outputs/topics_pos.csv')

df_n = pd.DataFrame.from_dict(n_date_dis, orient = 'index')
df_n = df_n.fillna(0)
df_n.to_csv('./outputs/topics_neg.csv')



df_p_freq = pd.DataFrame.from_dict(p_date_dis_freq, orient = 'index')
df_p_freq = df_p_freq.fillna(0)
df_p_freq.to_csv('./outputs/topics_pos_freq.csv')

df_n_freq = pd.DataFrame.from_dict(n_date_dis_freq, orient = 'index')
df_n_freq = df_n_freq.fillna(0)
df_n_freq.to_csv('./outputs/topics_neg_freq.csv')



df_total = pd.DataFrame.from_dict(total_date_dis)
df_total = df_total.fillna(0)
df_total.to_csv('./outputs/topics_total.csv')

df_rate = pd.DataFrame.from_dict(rate_date_dis)
df_rate = df_rate.fillna(0)
df_rate.to_csv('./outputs/topics_frequency.csv')

