# Project Details
## Spark Project: Sparkify
### Introduction

You'll learn how to manipulate large and realistic datasets with Spark to engineer relevant features for predicting churn. You'll learn how to use Spark MLlib to build machine learning models with large datasets, far beyond what could be done with non-distributed technologies like scikit-learn.

Predicting churn rates is a challenging and common problem that data scientists and analysts regularly encounter in any customer-facing business. 
Additionally, the ability to efficiently manipulate large datasets with Spark is one of the highest-demand skills in the field of data.

### Motivation/Essential Learnings

	* Load large datasets into Spark and manipulate them using Spark SQL and Spark Dataframes
	* Use the machine learning APIs within Spark ML to build and tune models

### Your Tasks
Your project will be divided into the following tasks

I. Load and clean the dataset

Load and clean the dataset (for example, records without userids or sessionids) so that you can further use it for the exploratory analysis and feature engineering part.

II. Define the label/user churn

Once you've done some preliminary analysis, create a column Churn to use as the label for your model. 
I suggest using the Cancellation Confirmation events to define your churn, which happen for both paid and free users. 
As a bonus task, you can also look into the Downgrade events.

III. Exploratory Data Analysis

Once the churn is defined, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. 
A good starting point would be by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

IV. Feature Engineering

Once you've familiarized yourself with the data, build out the features you find promising to train your model on. 
To work with the full dataset, you can follow the following steps.

	* Write a script to extract the necessary features from the smaller subset of data
	* Ensure that your script is scalable, using the best practices discussed in Lesson 3
	* Try your script on the full data set, debugging your script if necessary

V. Train models and tune the best model

Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. 
Evaluate the accuracy of the various models, tuning parameters as necessary. 
Determine your winning model based on test accuracy and report results on the validation set. 
Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

