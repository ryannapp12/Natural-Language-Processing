Name -> 
HomeWork Number -> 

Instructions to train and test:

   For features.py:
    python3 features.py

    Run this command to get output of the the features extraction function for the output

   For hw1.py: (Example)
    python3 hw1.py --train data/train.csv --test data/dev.csv --model Ngram+Lex --lexicon_path lexica/Sentiment140-Lexicon/Emoticon-bigrams.txt
    
    This command will train and test the model and prints out macro-averaged F1 score and classwise F1 accuracy

Special Features:
    I have used elongated text feature encoding as the special features implemented a function as elongated_words_encodings to get
    output of words having a char more that 2 times and saving it as encoding and using it for training model.