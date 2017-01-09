# yelpRatePrediction
This is a machine learning project using sklearn,scikit learn and nltk to predict user rating based on user review.

Yelp dataset is an academic dataset that contains information about local business. the dataset contains 5 json files
including business,checkin,reviews,tip and user.

this project focus in studying review file. the file contains user reviews about several businesses.

the yelp_academic_dataset_review.json contains several records , where each record contains encrypted user_id , encrypted 
business_id , user review as text and user rating (from 1 to 5).

we want to predict the user rating about specific business based on user review. the problem is simple :

    Given a user review , predict the user ratings.

this is machine learning classification algorithm where the input is text string and the output is number.

we should predict the output as a number from 1-5. to make it simple , I decide to make the problem as binary 
classification, trying to predict if the user likes this service or not. simply if the user select rate
from 1-3 , this means that the user don't like the service , where rating from 4-5 means that the user likes 
the service. so the problem become :

    Given a user review, predict if user like the service or not.
    
the project contains the following files:

preprocssing.py: python script to extract specific number of records from the rating file., to run this script , user
should pass the folder path of the dataset and the required number of records. for example to extract the first 2000
records , the user should run the following command.

    $python preprocssing.py [path to dataset file] [number of records to extract]. for example:

    $python preprocssing.py /Users/amrkoura/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json 2000

by running this command the scipt will generate two files: ratings.txt and reviews.txt which will be used later by
TextClassification.py

The second python script is TextClassification.py. user can run the script simply by:

    $python TextClassification.py
    
The script will read the data from ratings.txt and reviews.txt and start to do learn text classification.
I have followed the awesome nltk tutorial from this web site:

    https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/
    
The script will use differernt classification algorithms like support vector machine, logisitc regression , 
stochastic gradient descent,multinomial Naive Bayes , Bernoulli Naive Bayes and combine the result from these 
algorithms using majority vote.



