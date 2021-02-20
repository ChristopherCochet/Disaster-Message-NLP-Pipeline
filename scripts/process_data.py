import sys
import pandas as pd
from sqlalchemy import create_engine


def extract_val(string):
    '''
    extract_val - helper funtion to process a string with dashes
    input: the string to be processed
    output: first eleemnet of the string returned in the form of an int  
    '''
    return int(string.split('-')[1])

def load_data(messages_filepath, categories_filepath):
    '''
    load_data - reads the raw csv files and load these in dataframes and merges these
    input: two csv files and tyheir paths : message csv and catgeories csv 
    output: merge dataframe
    '''

    print("load_data - loading data files {} {}".format(messages_filepath, categories_filepath))
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # clean data
    print("load_data - cleaning data ...")
    col_list = categories_df['categories'].head(1).str.split(';').tolist()[0]
    col_list = [col.split('-')[0] for col in col_list]
    
    categories_expanded_df = categories_df['categories'].str.split(';',expand=True)
    categories_expanded_df.columns = col_list

    categories_expanded_df = categories_expanded_df.applymap(lambda x: extract_val(x))
    categories_expanded_df['id'] = categories_df['id']

    df = merge_data(messages_df, categories_expanded_df)

    return(df)


def merge_data(df1, df2):
    '''
    merge_data - inner join of two dataframes
    input: two dataframes
    output: teo dataframes inner joined on the id index  
    '''    
    print("merge_clean_data - dataframe {} {}".format(df1.shape, df2.shape))
    
    # Merge dataframes
    df = pd.merge(df1,df2, how = 'inner' , on = 'id')
    
    # Binarize
    mask = (df['related'] == 2)
    df.loc[mask, 'related'] = 1
    
    # Remove duplicates
    df.drop_duplicates(inplace = True)    

    return df


def save_data(df, database_filename):
    '''
    save_data - saves a dataframe to a sql lite db file and table
    input: dataframe and the sql lite db file
    output: N/A
    '''        
    print("save_data - write dataframe to database {}".format(database_filename))
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

    # Check Connection and table
    check_query_df = pd.read_sql_query("SELECT * FROM messages", con = engine).head()
    print("save_data - checking database access and table messages {}".format(check_query_df.shape))    


def main():
#   To run ETL pipeline that cleans data and stores in database
#       `python scripts/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()