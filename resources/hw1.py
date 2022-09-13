import argparse
from sklearn.svm import SVC
from sklearn.preprocessing import Binarizer, StandardScaler
from features import *
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Function to Combine 2 dataframes
def combine(df1, df2):
    final = df1.copy()
    keys1 = df1.keys()
    mapped = {}
    for x in keys1:
        mapped[x] = 1
    keys2 = df2.keys()
    for x in keys2:
        if mapped.get(x) == None:
            final[x] = df2[x]
    return final 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=False,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()
    path_to_train_data = args.train
    path_to_test_data = args.test
    model = args.model
    path_to_lexicon = args.lexicon_path
    df = pd.read_csv(path_to_train_data)
    test_df = pd.read_csv(path_to_test_data)

    #If model == Ngram use word ngrams
    if model == "Ngram":
        X = word_ngrams_features(path_to_train_data)
        Y = df["label"]
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)
        test = test_df["tweet_tokens"]
        X_test = pd.DataFrame()
        keys = X.keys()
        for key in keys:
            X_test[key] = test_df["tweet_tokens"].str.count(key)
        Y_pred = clf.predict(X_test.values)
        Y_true = test_df["label"]
        print("Macro averaged F1 score of the model")
        print(f1_score(Y_true, Y_pred, average='macro'))
        target_classes = ["positive", "negative", "objective", "neutral"]
        print("F1 score for each class separately")
        print(classification_report(Y_true, Y_pred, target_names=target_classes))
    
    #If model == Ngram+Lex use word ngrams + lexicon
    elif model == "Ngram+Lex":
        X2 = lexicon_features(path_to_lexicon, path_to_train_data)
        X1 = word_ngrams_features(path_to_train_data)
        Y = df["label"]
        X = combine(X1, X2)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)
        X_test = pd.DataFrame()
        keys = X.keys()
        count = 0
        for key in keys:
            if count == len(X1.keys()):
                break
            X_test[key] = test_df["tweet_tokens"].str.count(key)
            count += 1
        X_test = combine(X_test, lexicon_features(path_to_lexicon, path_to_test_data))
        Y_pred = clf.predict(X_test.values)
        Y_true = test_df["label"]
        print("Macro averaged F1 score of the model")
        print(f1_score(Y_true, Y_pred, average='macro'))
        target_classes = ["positive", "negative", "objective", "neutral"]
        print("F1 score for each class separately")
        print(classification_report(Y_true, Y_pred, target_names=target_classes))
    #If model == Ngram+Lex+Enc use word ngrams + Lexicon + 2 encodings
    elif model == "Ngram+Lex+Enc":
        X2 = lexicon_features(path_to_lexicon, path_to_train_data)
        X1 = word_ngrams_features(path_to_train_data)
        X3 = all_caps_encoding(path_to_train_data)
        X4 = postag_encoding(path_to_train_data)
        X_temp = combine(X1, X2)
        X_temp = combine(X_temp, X3)
        X = combine(X_temp, X4)
        Y = df["label"]
        mapped = {}
        for key in X4.keys():
            mapped[key] = 1
        keys = X.keys()
        X_test = pd.DataFrame()
        for key in keys:
            if type(key) == int:
                X_test[key] = X[key]
                continue
            if mapped.get(key) == None:
                X_test[key] = test_df["tweet_tokens"].str.count(key)
            else:
                X_test[key] = test_df["pos_tags"].str.count(key)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)
        Y_pred = clf.predict(X_test.values)
        Y_true = test_df["label"]
        print("Macro averaged F1 score of the model")
        print(f1_score(Y_true, Y_pred, average='macro'))
        target_classes = ["positive", "negative", "objective", "neutral"]
        print("F1 score for each class separately")
        print(classification_report(Y_true, Y_pred, target_names=target_classes))

    #If model == Custom use word ngrams + Lexicon + 3 encodings
    elif model == "Custom":
        X2 = lexicon_features(path_to_lexicon, path_to_train_data)
        X1 = word_ngrams_features(path_to_train_data)
        X3 = all_caps_encoding(path_to_train_data)
        X4 = postag_encoding(path_to_train_data)
        X5 = elongated_words_encodings(path_to_train_data)
        X_temp = combine(X1, X2)
        X_temp = combine(X_temp, X3)
        X_temp = combine(X_temp, X5)
        X = combine(X_temp, X4)
        Y = df["label"]
        mapped = {}
        for key in X4.keys():
            mapped[key] = 1
        keys = X.keys()
        X_test = pd.DataFrame()
        for key in keys:
            if type(key) == int:
                X_test[key] = X[key]
                continue
            if mapped.get(key) == None:
                X_test[key] = test_df["tweet_tokens"].str.count(key)
            else:
                X_test[key] = test_df["pos_tags"].str.count(key)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)
        Y_pred = clf.predict(X_test.values)
        Y_true = test_df["label"]
        print("Macro averaged F1 score of the model")
        print(f1_score(Y_true, Y_pred, average='macro'))
        target_classes = ["positive", "negative", "objective", "neutral"]
        print("F1 score for each class separately")
        print(classification_report(Y_true, Y_pred, target_names=target_classes))
