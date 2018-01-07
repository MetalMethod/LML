"""
Igor Busquets LML
"""
#Natural Language Processing

#####GENERAL COMMENTS FROM THE VIDEO
#Tokenization words = one column for each word
#Bag of Words = only the relevant distinct words, ignoring pontuantion and apply stemming, with is add LOVE but not LOVED or LOVING, stemming is adding only LOVE
#sparse matrix, because most of the words will not be in the reviews
#sparse matrix = a matrix with a lot of zeros

#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# re is for cleaning text
import re

#NLTK: NLP  tratment lib
# stopwords list is the ignored words list

#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#stemming class
from nltk.stem.porter import PorterStemmer

#Tolkenization class
from sklearn.feature_extraction.text import CountVectorizer

#Import the dataset
#set delimiter to tab character and quoting = 3 is for igonre double quotes
dataset = pd.read_csv('Restaurant_reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the text for NLP
#aplly cleaning to one review (row)only , later iterate to all the rows

#show first row of dataset
dataset['Review'][0]

#corpus = common word for collection of text of the same type
corpus = []
#The loop is for iteration all the reviews and transform each one
for i in range(0, 1000):
    ###########FIRST STEP: KEEP ONLY THE DIFFERENT LETTERS A TO Z sub(regex, add space between removed chars, text)
    #cleanined review = review
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] )
    ###########SECOND STEP:  put all the letters in lowercase
    review = review.lower()
    ###########THIRD STEP: remove the non significant words (the , this, then that...)
    # list of different words for each review
    review = review.split()
    #iterate to all words in each review
    #but only those NOT in the english list of stopwrods
    #in a set, python iterations goes way faster
    # the third and forth step are implemented in the same line below
    ###########FOURTH STEP: STEMMING, remove variations of words, keeeping the root of that word
    #done to avoid sparsinity
    ps = PorterStemmer()
    # STEP 3 AND 4 in the same line
    #step 4 is only ps.stem(word) 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #FIFTH STEP: Revert the transofrmation of spliited list for review to a single review with stemmed words
    #adding spaces between words
    review = ' '.join(review)
    #add cleanined review to the corpus
    corpus.append(review)
    
#CREATING THE BAG OF WORDS BY TOKENIZATION = sparse matrix of features
#Ignoring the words that appears only a very few of times
# creating a column for each word and a number that represents the number of times that word appeared in each of the reviews
#requires a dependent variable vector for classification (column Liked with 0 or 1)
#the model will predict that independt variable
#matrix of features = bag of words

#from sklearn.feature_extraction.text import CountVectorizer
# max_features = 1500 keeps only the first 1500 words with more presence, igonoring the rest 
cv = CountVectorizer(max_features = 1500)

#create the big sparse matrx that is a matrix of features X, thats why theres toarray()
X = cv.fit_transform(corpus).toarray()

#dependent variable (vector y)
#select the second column of the dataset, the liked column
y = dataset.iloc[:, 1].values

############# TRAINING

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,  y_train)
'''

# MAXIMUM ENTROPY CLASSIFIER
import nltk.classify.maxent
encoding = nltk.classify.maxent.MaxentFeatureEncodingI

model = nltk.classify.maxent.MaxentClassifier(encoding, weights, logarithmic=True)


########Prediction X_test set using classifier
# vector of each prediction
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test ,y_pred)

##### MODEL EVATUATION
#Confusion matrix values
true_negatives = cm[0,0]
true_positives = cm[1,1]
false_negatives = cm[0,1]
false_positives = cm[1,0]

total_preds = 0
for pred in np.nditer(cm):
        total_preds = total_preds + pred
        
#accuracy in percentage
if total_preds != 0:
    accuracy = (true_negatives + true_positives) / total_preds
    #print("total true prediction: ", (true_negatives + true_positives))
    print("accuracy: ", accuracy*100, "%")
    
#Precision = TP / (TP + FP)
precision = true_positives / (true_positives + false_positives)
print("precision: ", "{:.6f}".format(precision))

#Recall = TP / (TP + FN)
recall = true_positives / (true_positives + false_negatives)
print("recall: ", "{:.6f}".format(recall))

#F1 Score = 2 * Precision * Recall / (Precision + Recall)
f1score = 2 * precision * recall / (precision + recall)
print("F1 score: " , "{:.6f}".format(f1score))

