#    - To run ML pipeline that trains classifier and saves
#        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

def load_data(database_filepath):
    # load data from database
    print("load_data - loading file {}".format(database_filepath))

    engine = create_engine('sqlite:///'+database_filepath)

    df = pd.read_sql_query("SELECT * FROM messages", con = engine)
    df = df.set_index('id')
    X = df.drop('genre', axis = 1)
    Y = df.iloc[:,2]
    
    return X, Y, df['genre'].unique()


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_training_set, Y_training_set):

    nlp_pipeline_simple_lgb = Pipeline([
                        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
                        ('multiclassifier',LGBMClassifier(n_jobs=-1))
                    ])

    print("build_model - model {} on X_train {} and Y_train {}".format(nlp_pipeline_simple_lgb, X_training_set.shape,Y_training_set.shape))                    

    myscoring = make_scorer(f1_score,average='weighted')

    parameters = {
        'tfidfvect__ngram_range': ((1, 1), (1, 2)),
        'tfidfvect__max_df': (0.5, 0.75, 1.0),
        'tfidfvect__max_features': (None, 100, 500, 2000),
        'multiclassifier__learning_rate': [0.01, 0.1 , 0.5, 0,8],
        'multiclassifier__max_depth': [5, 10, 20],
        'multiclassifier__n_estimators': [100, 200, 400, 800 , 1000] 
        }

    # create grid search object
    search = RandomizedSearchCV(nlp_pipeline_simple_lgb, parameters, scoring=myscoring, n_jobs=-1)
    result = search.fit(X_training_set, Y_training_set)

    # summarize result
    print('"build_model - best Score: %s' % result.best_score_)
    print('"build_model - best Hyperparameters: %s' % result.best_params_)

    return(search.best_estimator_)


def evaluate_model(model, X_test, Y_test, category_names):
    print("evaluate_model - model {}".format(model)) 
    
    predictions = model.predict(X_test)
    cm = confusion_matrix(Y_test, predictions)
    print("evaluate_model - confusion matrix {}".format(cm))
    print(classification_report(Y_test, predictions, labels = model.classes_))



def save_model(model, model_filepath):

    #Get the date to time stamp model
    print("save_model - model {} to {}".format(model, model_filepath)) 

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print("X -> {} Y -> {} Category - {}".format(X.shape, Y.shape,category_names))
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building & Training model...')
        model = build_model(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test['message'], Y_test, category_names)

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