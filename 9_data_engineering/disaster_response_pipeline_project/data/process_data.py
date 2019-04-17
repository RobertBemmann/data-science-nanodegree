import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""Loads 2 csv files and merges them into one dataframe.
    
	Args:
	messages_filepath: the filepath of the messages csv file
	categories_filepath: the filepath of the categories csv file 
	
	Returns:
	df: the merged dataframe of the 2 csv files
	"""
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	df = messages.merge(categories, on='id')
	
	return df

def clean_data(df):
    """Loads, cleans and returns the dataframe for the ML pipeline.
    """
    # Split categories into separate category columns.
    categories = df[['id','categories']]
    row = categories.iloc[0]
    category_colnames = row.categories.split(';')
    category_colnames = list(map(lambda x: str(x)[:-2], category_colnames))
    helper = pd.DataFrame(columns=category_colnames)
    categories = categories.join(helper)
    categories.drop(['categories'], axis=1, inplace=True)
	
    # Convert category values to just numbers 0 or 1
    for column in category_colnames:
    # set each value to be the last character of the string
        start = df.categories.astype(str).str.find(column)[0]
        start += len(str(column))+1
        categories[column] = df.categories.astype(str).str[start:start+1]
    
		# convert column from string to numeric and all values == 2 to 1
        categories[column] = pd.to_numeric(categories[column], downcast='integer').replace(2, 1)

	# drop the original categories column from 'df'
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new 'categories' dataframe
    df = df.merge(categories, on='id')
	
    # drop duplicates
    df.drop_duplicates(subset ='message', keep = 'first', inplace = True) 
	
    return df
	
def save_data(df, database_filename):
    """Loads dataframe and saves it into database file.	
	"""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_tagged', engine, index=False)
	
def main():
    """This function executes an end to end ETL pipeline.
	
	The function loads the data from 2 csv files and merges them
	into a dataframe. Next, the dataframe is cleansed and prepared
	for the ML pipeline. Eventually, the cleansed dataframe gets
	stored in a SQLite database file.
	
	Args:
	messages_filepath: the filepath of the messages csv file
	categories_filepath: the filepath of the categories csv file
	database_filepath: the filepath where the SQLite database file
	shall get stored
	"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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