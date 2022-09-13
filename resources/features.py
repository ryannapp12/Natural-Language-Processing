from collections import OrderedDict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict, Counter
import re
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams

## Extracting char ngram features
def char_ngrams_features(path_to_data):
    df = pd.read_csv(path_to_data)
    tweets = df["tweet_tokens"]
    cv = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=3, max_df=.95, lowercase=True)
    features_char_ngrams = cv.fit_transform(tweets)
    features_char_ngrams_df = pd.DataFrame(features_char_ngrams.toarray(), index=tweets.index, columns=cv.get_feature_names())
    return features_char_ngrams_df

## Extracting word ngram features
def word_ngrams_features(path_to_data):
    df = pd.read_csv(path_to_data)
    tweets = df["tweet_tokens"]
    cv = CountVectorizer(ngram_range=(1,4), min_df=3, max_df=.95, stop_words='english')
    features_word_ngrams = cv.fit_transform(tweets)
    features_word_ngrams_df = pd.DataFrame(features_word_ngrams.toarray(), index=tweets.index, columns=cv.get_feature_names())
    return features_word_ngrams_df

# Extracting polarity
def polarity(x, wordDict):
    score = wordDict[x]
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    else:
        return 'none'

# Extracting count tokes with polarity
def count_tokens_with_polarity(string, tokenizer, wordDict):
    
    scorelist = []
    tokenized = tokenizer.tokenize(string)
    ngrams_list = [' '.join(i) for i in ngrams(tokenized, 2)]
    for ngram in ngrams_list:
        ngram = ngram.lower()
        score = polarity(ngram, wordDict)
        scorelist.append(score)
        
    return dict(Counter(scorelist))

# Polarity sum
def polarity_sum(string, tokenizer, wordDict):
    
    negList = []
    posList = []
    tokenized = tokenizer.tokenize(string)
    ngrams_list = [' '.join(i) for i in ngrams(tokenized, 2)]
    for ngram in ngrams_list:
        ngram = ngram.lower()
        if polarity(ngram, wordDict) == 'positive':
            posList.append(wordDict[ngram])
        elif polarity(ngram, wordDict) == 'negative':
            negList.append(abs(wordDict[ngram]))
        
    return {'pos_sum' : sum(posList), 'neg_sum' : sum(negList)}

# Maximum token with polarity
def max_token(string, tokenizer, wordDict):
    
    negList = []
    posList = []
    
    tokenized = tokenizer.tokenize(string)
    ngrams_list = [' '.join(i) for i in ngrams(tokenized, 2)]
    for ngram in ngrams_list:
        ngram = ngram.lower()
        if polarity(ngram, wordDict) == 'positive':
            posList.append(wordDict[ngram])
        elif polarity(ngram, wordDict) == 'negative':
            negList.append(wordDict[ngram])
        
        
    try:
        pos_max = max(posList)
    except ValueError:
        pos_max = 0
    try:
        neg_max = min(negList)
    except ValueError:
        neg_max = 0
        
    return {'pos_max' : pos_max, 'neg_max' : neg_max}

# Last token
def last_token(string, tokenizer, wordDict):
    
    tokenized = tokenizer.tokenize(string)
    ngrams_list = [' '.join(i) for i in ngrams(tokenized, 2)]
    
    for token in reversed(ngrams_list):
        token = token.lower()
        if polarity(token, wordDict) == 'positive' or polarity(token, wordDict) == 'negative':
            return {'last_polarity' : wordDict[token]}
        else:
            continue
    
    return {'last_polarity' : 0}

# All features using a lexicon (neg_max, positive, polarity, negative)
def all_feats_dict(string, tokenizer, wordDict):
    ct = count_tokens_with_polarity(string, tokenizer, wordDict)
    pol = polarity_sum(string, tokenizer, wordDict)
    max_tkn = max_token(string, tokenizer, wordDict)
    last = last_token(string, tokenizer, wordDict)
    
    complete = dict()
    
    for dictionary in [ct, pol, max_tkn, last]:
        complete.update(dictionary)
        
    return complete

# Lexicon features extraction
def lexicon_features(lexicon_path, path_to_data):
    df = pd.read_csv(path_to_data)
    tweets = df["tweet_tokens"]
    wordDict = defaultdict(float)
    with open(lexicon_path, 'r') as f:
        for row in f.readlines():
            row = row.split()
            if len(row) == 5:
                wordDict[row[0] +" " + row[1]] = float(row[2])
            else:
                wordDict[row[0]] = float(row[1])
    tokenizer = TweetTokenizer()
    lexicon_feature_counts = [all_feats_dict(tweet, tokenizer, wordDict) for tweet in tweets]
    lexicon_features_df = pd.DataFrame(lexicon_feature_counts, index=tweets.index)
    lexicon_features_df = lexicon_features_df.fillna(0)
    return lexicon_features_df

## All caps encoding
def all_caps_encoding(path_to_data):
    df = pd.read_csv(path_to_data)
    all_caps_encoding = df['tweet_tokens'].str.count(r'[A-Z]{2,}')
    all_caps_encoding.name = 'ct_allcaps'
    return all_caps_encoding

## Hashtag count encoding
def hashtag_count_encoding(path_to_data):
    df = pd.read_csv(path_to_data)
    alltext = ' '.join([i for i in df['tweet_tokens']])
    hashtag_vocabulary = list(set(re.findall(r'\#\w+', alltext)))
    hashtag_encoding = pd.DataFrame()
    for hashtag in hashtag_vocabulary:
        hashtag_encoding[hashtag] = df['tweet_tokens'].str.count(hashtag)
    return hashtag_encoding

## Postag encoding
def postag_encoding(path_to_data):
    df = pd.read_csv(path_to_data)
    df["pos_tags"] = str(df["pos_tags"])
    alltext = ' '.join([i for i in df['pos_tags']])
    postag_vocabulary = list(set(list(alltext.split())))
    postag_encoding = pd.DataFrame()
    for postag in postag_vocabulary:
        postag_encoding[postag] = df['pos_tags'].str.count(postag)
    return postag_encoding

## Elongated words encodings
def elongated_words_encodings(path_to_data):
    df = pd.read_csv(path_to_data)
    alltext = ' '.join([i for i in df["tweet_tokens"]])
    elongated_words_vocabulary = []
    for word in list(alltext.split()):
        mapped = {}
        flag = 0
        for i in range(len(word)):
            if not ((word[i] >= 'a' and word[i] <= 'z') or (word[i] >= 'A' and word[i] <= 'Z')):
                flag = 1
                break
            if mapped.get(word[i]) == None:
                mapped[word[i]] = 1
            else:
                mapped[word[i]] += 1
        if flag == 1:
            continue
        for (c, count) in mapped.items():
            if count > 2 and ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')):
                elongated_words_vocabulary.append(word)
                break
    elongated_words_vocabulary = list(set(elongated_words_vocabulary))
    elongated_words_encoding = pd.DataFrame()
    for elongated_word in elongated_words_vocabulary:
        elongated_words_encoding[elongated_word] = df["tweet_tokens"].str.count(elongated_word)
    return elongated_words_encoding

if __name__ == "__main__":
    print("Features of training data using char ngrams:")
    print(char_ngrams_features("data/train.csv").head(10))
    print("Features of training data using word ngrams:")
    print(word_ngrams_features("data/train.csv").head(10))
    print("Features of training data using Sentiment120 bigram lexicon:")
    print(lexicon_features("lexica/Sentiment140-Lexicon/Emoticon-bigrams.txt", "data/train.csv").head(10))
    print("Features of training data using All Caps Encoding:")
    print(all_caps_encoding("data/train.csv").head(10))
    print("Features of training data using Hashtag Count Encoding:")
    print(hashtag_count_encoding("data/train.csv").head(10))
    print("Features of training data using Postag Encoding:")
    print(postag_encoding("data/train.csv").head(10))
    print("Features of training data using Elongated word Encoding:")
    print(elongated_words_encodings("data/train.csv"))
