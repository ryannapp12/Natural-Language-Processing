NAME: Ryan Napolitano
EMAIL: rn2473@columbia.edu
Homework 1

How to test classifier:
You can just run the file and include some of the features I have implemented into x_train and x_test.
Example:

elif model == 'Custom':
        cn, vect = char_ngram_extraction(df_train)
        ewe = elongated_words_encoding(df_train)
        x_train = hstack((cn, ewe))
        #x_train = (cn)     # comment or uncomment these lines or add features

        clf = MultinomialNB()
        clf.fit(x_train, y_train)

        cn = vect.transform(df_test.tweet_tokens)
        ewe = elongated_words_encoding(df_test)
        x_test = hstack((cn, ewe))
        #x_test = (cn)       # comment or uncomment these lines or add features




