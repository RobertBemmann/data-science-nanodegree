# Disaster Response Pipeline Project

### Project Description:
	This project contains a Flask Web App (Python) which uses a trained Machine Learning model to categorize input messages relaeted to disasters.
    The Web App has a main page where you can enter a message. If you click on 'Classify Message', you will be directed to a page that shows
	the classification prediction according to the underlying model. Before you can run the Web App, you can execute an ETL and a Machine Learning
	pipeline in order to get a trained model that serves the app.

### File Structure:

```
- app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

- data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl # saved model

- README.md
```

### File Descriptions:
	1. process_data.py - an ETL Pipeline that:
	* Loads the messages and categories datasets
	* Merges the two datasets
	* Cleans the data
	* Stores it in a SQLite database

	2. train_classifier.py - a ML Pipeline that:
	* Loads data from the SQLite database
	* Splits the dataset into training and test sets
	* Builds a text processing and machine learning pipeline
	* Trains and tunes a model using GridSearchCV
	* Outputs results on the test set
	* Exports the final model as a pickle file

    3. run.py - Flask Web App
	* Modify file paths for database and model as needed
	* Add data visualizations using Plotly in the web app. One example is provided for you

### How to Interact with the project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or localhost:3001
