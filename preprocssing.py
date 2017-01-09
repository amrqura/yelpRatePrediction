'''
Written By Amr Koura
In this python script I will solve the newyorker data challenge.
The task is very wide , it is an open ended task.
The task is to download the yelp dataset , study it and try to formulate
interesting questions and try to solve them using any machine learning library
In this script I will solve the following question
Q:
'''
import json
import sys
def main():
    data_set_folder=sys.argv[1]
    file_limit=int(sys.argv[2])
    user_reviews=[]
    user_ratings=[]
    data=[]
    count=1
    with open(data_set_folder+'/yelp_academic_dataset_review.json') as f:    
        for line in f:
           tmp=json.loads(line)
           user_reviews.append(tmp['text'])
           user_ratings.append(tmp['stars'])
           count=count+1
           if count>file_limit:
               break 
    import pickle
    print 'Saving reviews'
    with open('reviews.txt', 'wb') as fp:
        pickle.dump(user_reviews, fp)
    print 'saving ratings'    
    with open('ratings.txt', 'wb') as fp:
        pickle.dump(user_ratings, fp)    
    print 'Done !!'       

if __name__ == '__main__':
    main() 