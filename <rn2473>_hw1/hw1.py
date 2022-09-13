import argparse
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from features import *
from scipy.sparse import hstack



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()

    model = args.model
    path_to_training_file = args.train
    path_to_test_file = args.test


    # read in the training data (tweets)
    df_train = pre_processed_data(path_to_training_file)
    y_train = df_train.label

    # read in the test data
    df_test = pre_processed_data(path_to_test_file)
    y_test = df_test.label

    # extract features into x_train, create and fit classifier(clf), transform test data into x_test
    clf = None
    x_train = x_test = None
    if model == 'Ngram':
        x1, vect = word_ngram_extraction(df_train)
        x_train = x1

        clf = MultinomialNB()
        clf.fit(x_train, y_train)

        x_test = vect.transform(df_test.tweet_tokens)

    # TODO: Implement Lexicon Model. Same result as Ngram
    elif model == 'Lex':
        x1, vect = word_ngram_extraction(df_train)
        x_train = x1

        clf = MultinomialNB()
        clf.fit(x_train, y_train)

        x_test = vect.transform(df_test.tweet_tokens)

    # Included Word Ngrams and All Caps Encoding, and Hashtag Counts Encoding.
    # Does not yet include the lexicon features.
    elif model == 'Ngram+Lex+Enc':
        wn, vect = word_ngram_extraction(df_train)
        hce = hashtag_count_encoding(df_train)
        ace = all_caps_encoding(df_train)
        x_train = hstack((wn, hce, ace))

        clf = MultinomialNB()
        clf.fit(x_train, y_train)

        wn = vect.transform(df_test.tweet_tokens)
        hce = hashtag_count_encoding(df_test)
        ace = all_caps_encoding(df_test)
        x_test = hstack((wn, hce, ace))

    # Uses char Ngram and Enlongated Words Encoding
    elif model == 'Custom':
        cn, vect = char_ngram_extraction(df_train)
        ewe = elongated_words_encoding(df_train)
        x_train = hstack((cn, ewe))
        #x_train = (cn)

        clf = MultinomialNB()
        clf.fit(x_train, y_train)

        cn = vect.transform(df_test.tweet_tokens)
        ewe = elongated_words_encoding(df_test)
        x_test = hstack((cn, ewe))
        #x_test = (cn)

    else:
        print("Error: No Such Model - " + model)
        exit()

    # Cross-validate
    cv = cross_validate(clf, x_train, y_train)
    print(cv)

    # Predict and evaluate
    y_predicted = clf.predict(x_test)
    print("\nModel: " + model)
    print('Classification report:')
    print(metrics.classification_report(y_test, y_predicted))






