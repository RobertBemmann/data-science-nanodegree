# Project Details
## Spark Project: Sparkify

### Link to Medium article

https://medium.com/@robertbemmann/predicting-user-churn-for-a-music-streaming-app-with-spark-pyspark-a4b4001fda22

### Introduction

You'll learn how to manipulate large and realistic datasets with Spark to engineer relevant features for predicting churn. You'll learn how to use Spark MLlib to build machine learning models with large datasets, far beyond what could be done with non-distributed technologies like scikit-learn.

Predicting churn rates is a challenging and common problem that data scientists and analysts regularly encounter in any customer-facing business. 
Additionally, the ability to efficiently manipulate large datasets with Spark is one of the highest-demand skills in the field of data.

### Motivation/Essential Learnings

 * Load large datasets into Spark and manipulate them using Spark SQL and Spark Dataframes
 * Use the machine learning APIs within Spark ML to build and tune models

### Tasks/Walkthrough

`I. Load and clean the dataset`

Load and clean the dataset (for example, records without userids or sessionids) so that you can further use it for the exploratory analysis and feature engineering part.

`II. Define the label/user churn`

Once you've done some preliminary analysis, create a column Churn to use as the label for your model. 
I suggest using the Cancellation Confirmation events to define your churn, which happen for both paid and free users. 
As a bonus task, you can also look into the Downgrade events.

`III. Exploratory Data Analysis`

Once the churn is defined, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. 
A good starting point would be by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

`IV. Feature Engineering`

Once you've familiarized yourself with the data, build out the features you find promising to train your model on. 
To work with the full dataset, you can follow the following steps.

 * Write a script to extract the necessary features from the smaller subset of data
 * Ensure that your script is scalable, using the best practices discussed in Lesson 3
 * Try your script on the full data set, debugging your script if necessary

`V. Train models and tune the best model`

Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. 
Evaluate the accuracy of the various models, tuning parameters as necessary. 
Determine your winning model based on test accuracy and report results on the validation set. 
Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.

### Results
I tried 3 ML classifiers:
* LogisticRegression
* NaiveBayes
* RandomForestClassifier

After training and evaluating the classifiers, it turned out that the RandomForestClassifier had the best F1 Score of 0.78 for the default parameters.
Therefore, I tuned this classifier with the CrossValidator and 6 possible parameter combinations.

The parameters for the CrossValidator were:
* maxDepth - [1, 3, 10]
* maxBins - [8, 16]

The best model identified so far was:
* RandomForestClassificationModel with 10 trees


### Libraries
* pyspark.sql module
  * SparkSession, types, functions, Window
* pyspark.ml package
  * tuning
  * regression
  * classification
  * feature
  * evaluation
* pyspark_dist_explore
* numpy
* pandas
* matplotlib.pyplot
* datetime

### List of files
* Data_Wrangling_and_Exploration (Jupyter Notebook and html-version)
  * includes the data loading, cleansing and labeling as well as the data exploration part with the plots
* Feature_Engineering_and_Model_Training (Jupyter Notebook and html-version)
  * includes the feature engineering part as well as model training, evaluation and tuning