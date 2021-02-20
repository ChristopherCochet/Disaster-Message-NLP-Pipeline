import sys
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, plot_importance
import re
import nltk
import pickle

from sqlalchemy import create_engine
from datetime import datetime
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

def load_data(database_filepath):
    '''
    load_data - loads the messages data from the SQL lite DB file to a dataframe
                and return the features, target and target labels
    input: SQL lite DB file
    output: features dataframe, target dataframe and target labels list
    '''
    print("load_data - loading file {}".format(database_filepath))

    engine = create_engine('sqlite:///'+database_filepath)

    df = pd.read_sql_query("SELECT * FROM messages", con = engine)
    # print(df.head())
    df = df.set_index('id')
    X = df['message']
    Y = df.iloc[:,3:]
    
    return X, Y, Y.columns.to_list()


def tokenize(text):
    '''
    tokenize - tokenize a string
    input: the string to be processed and tokenized
    output: the tokens
    '''    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_training_set, Y_training_set):
    '''
    build_model - trained and tune an nlp pipeline
    input: Training dataset and labels
    output: nlp pipeline
    ''' 

    nlp_chain_nlp_pipeline_lgb = Pipeline([
                            ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
                            ('classifierchain',ClassifierChain(LGBMClassifier(n_jobs=-1)))
                        ])

    print("build_model - model {} \n\n on X_train {} and Y_train {}".format(nlp_chain_nlp_pipeline_lgb, X_training_set.shape, Y_training_set.shape))                    

    myscoring = make_scorer(f1_score,average='weighted')

    parameters = {
     'tfidfvect__ngram_range': ((1, 1), (1, 2)),
     'tfidfvect__max_df': (0.5, 0.75, 1.0),
     'tfidfvect__max_features': (None, 100, 500, 2000)
    }

    # create grid search object
    search = RandomizedSearchCV(nlp_chain_nlp_pipeline_lgb, parameters, scoring=myscoring, n_jobs=-1)
    result = search.fit(X_training_set, Y_training_set)

    # summarize result
    print('"build_model - best Score: %s' % result.best_score_)
    print('"build_model - best Hyperparameters: %s' % result.best_params_)

    return(search.best_estimator_)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model - display classification report of the pipeline provided
    input: pipeline, traing and label sets as well as the labels
    output: N/A
    '''    

    print("evaluate_model - model {}".format(model)) 
    
    predictions = model.predict(X_test)
    print("evaluate_model - classification report")
    print(classification_report(Y_test, predictions, target_names = Y_test.columns.to_list()))


def save_model(model, model_filepath):
    '''
    save_model - dsave sthe trained pipleine and model to a pickle file
    input: pipleine and path to pickle file where the model will be saved
    output: N/A
    '''    

    print("save_model - model {} to {}".format(model, model_filepath)) 

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
#     To run ML pipeline that trains classifier and saves
#     `python scripts/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print("X -> {} Y -> {} Category - {}".format(X.shape, Y.shape, category_names))
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
        
        print('Building & Training model...')
        model = build_model(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()