# import libraries
import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import datetime
import pickle

def load_data(database_filepath):
	"""Loads dataframe and returns features, labels and label names.
	
	Args:
	database_filepath: the filepath of the SQLite database file
	
	Returns:
	X: array of dataframe features
	y: array of dataframe labels
	category_names: list of category names, matches indices of y
	"""
	engine = create_engine('sqlite:///'+database_filepath)
	df = pd.read_sql("SELECT * FROM messages_tagged", engine)
	X = df.message.values
	y = df.iloc[:,4:40].values
	category_names = df.iloc[:,4:40].columns
	return X, y, category_names

def tokenize(text):
    """NLP processing function of input messages.
	
	Args:
	text: the raw text messages
	
	Returns:
	clean_tokens: tokenized text messages 
	"""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Builds and returns a GridSearchCV ML pipeline.
	"""
    pipeline =Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters = {'tfidf__use_idf': (True, False), 'clf__estimator__n_estimators': (10, 20)}
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Validates the best ML model and prints a classification_report.
	
	Args:
	model: the trained ML model
	X_test: the features of the test set
	Y_test: the labels of the test set
	category_names: the label names of the test set
	"""
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Saves the best ML model as a pickle file.
	"""
    best_model = model.best_estimator_
    fileObject = open(model_filepath,'wb')
    pickle.dump(best_model,fileObject)
    fileObject.close()


def main():
    """Executes the ML pipeline and saves the best model as a pickle file.
	
	Args:
	database_filepath: the path where the database file is stored
	model_filepath: teh path where the model pickle file shall
	get stored
	"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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