from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import csr_matrix



'''** Word ngram features **'''
def word_ngram_extraction(df):
    cv = CountVectorizer(analyzer='word', ngram_range=(1, 4), min_df=3, max_df=.95)
    tweet_tokens = df.tweet_tokens
    word_ngram_matrix = cv.fit_transform(tweet_tokens)
    return word_ngram_matrix, cv

'''** Char ngram features **'''
def char_ngram_extraction(df):
    cv = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3, max_df=.95)
    tweet_tokens = df.tweet_tokens
    char_ngram_matrix = cv.fit_transform(tweet_tokens)
    return char_ngram_matrix, cv

'''** Hashtag Count Encoding **'''
def hashtag_count_encoding(df):
    tweet_tokens = df.tweet_tokens
    hashtag_counts = tweet_tokens.str.count(r'\#[A-Za-z][A-Za-z0-9_]*')
    return csr_matrix(hashtag_counts).T

'''** All Caps Encoding **'''
def all_caps_encoding(df):
    tweet_tokens = df.tweet_tokens
    all_caps_counts = tweet_tokens.str.count(r'\b[A-Z]{2,}\b')
    return csr_matrix(all_caps_counts).T

'''** Elongated Words Encoding **'''
def elongated_words_encoding(df):
    tweet_tokens = df.tweet_tokens
    ewe = tweet_tokens.str.count(r'\b[A-Za-z]*([A-Za-z])\1{3,}[A-Za-z]*\b')
    return csr_matrix(ewe).T

# Helper function to read in file
def pre_processed_data(csv_file):
    df = pd.read_csv(csv_file)
    df.loc[df.label == 'objective', 'label'] = 'neutral'
    return df


if __name__ == "__main__":
    path = "/Users/ryannapp/PycharmProjects/hw1(rn2473)/"
    df = pre_processed_data(path + "data/train.csv")

    print("Words ngrams: ")
    x, v = word_ngram_extraction(df)
    print(x.shape)
    print(x)

    print("Hashtag Count Encoding: ")
    x = hashtag_count_encoding(df)
    print(x.shape)
    print(x)

    print("All Caps Encoding: ")
    x = all_caps_encoding(df)
    print(x.shape)
    print(x)

    print("Elongated Words Encoding: ")
    x = elongated_words_encoding(df)
    print(x.shape)
    print(x)