# import packages
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk

def extract_val(string):
    return int(string.split('-')[1])

def load_data(data_file):
    # read in file
    print("load_data - loading file {}".format(data_file))
    messages_df = pd.read_csv(data_file)

    # clean data
    print("load_data - cleaning data ...")
    col_list = categories_df['categories'].head(1).str.split(';').tolist()[0]
    col_list = [col.split('-')[0] for col in col_list]
    
    categories_expanded_df = categories_df['categories'].str.split(';',expand=True)
    categories_expanded_df.columns = col_list

    categories_expanded_df = categories_expanded_df.applymap(lambda x: extract_val(x))
    categories_expanded_df['id'] = categories_df['id']

    df = pd.merge(messages_df,categories_expanded_df, how = 'inner' , on = 'id')
    df.drop_duplicates(inplace = True)
    print("load_data - dataframe {}".format(df.head()))

    # load to database
    db_name = 'disaster-message.db'
    print("load_data - write dataframe to database {}".format(db_name))
    engine = create_engine('sqlite:///data/'+db_name)
    df.to_sql('messages', engine, index=False)

    # define features and label arrays
    X = df.drop('genre', axis = 1)
    y = df.iloc[:,2]
    print("load_data - extract X {} & y {}".format(X.head(), y.head()))

    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    # model = build_model()  # build model pipeline
    # model = train(X, y, model)  # train model pipeline
    # export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
