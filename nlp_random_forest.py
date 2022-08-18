import re
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import string

# nltk.download()



stop_= stopwords.words('english')


data_path="../SMSSpamCollection.tsv"
data=pd.read_table(data_path,header=None)
data.columns=["label","text"]

# EXPLORE THE DATASET

# 1. Check the shape of the data
print(len(data))
print(len(data.columns))

# Check the distribution of the labels!
ham_num= data[data["label"]=="ham"]
spam_num= data[data["label"]=="spam"]


print(len(data))
print(len(spam_num))
print(len(ham_num))

# How much is there a missing data!
print(data["label"].isnull().sum())
print(data["text"].isnull().sum())



# 1. Punctiation
def remove_character(data):
    data['text_new'] = data['text'].str.replace(r'[^\w\s]+', '')
    return data
data= remove_character(data)


# 2 Make lower case and tokenization
def tokenizer(data):
    data["text_new"] = data["text_new"].str.lower()
    data["text_new"]=[re.split("\W+",i) for i in data["text_new"].to_list()]
    return data
data= tokenizer(data)


# Remove stop words
stopwords=nltk.corpus.stopwords.words("english")

def stop_remove(data,stopwords):
    # data['text_new'].apply(lambda x: [item for item in x if item not in stopwords])
    for ind in range(0,len(data)):
        data["text_new"].loc[ind]=[i for i in data["text_new"].loc[ind]if i not in stopwords]
    return data
data= stop_remove(data,stopwords)


# Test out the Porter Stemmer - There are many different number of stemmer, this is optional!
'''
1. It decrease the corpus of the words the model exposed to
2. It makes the words which have similiar meaning to be in the same form of the word.
'''


# ps = nltk.PorterStemmer()
#
# def stem_words(data):
#     for ind in range(0,len(data)):
#         data["text_new"].loc[ind]=[ps.stem(i) for i in data["text_new"].loc[ind]]
#     return data
# data= stem_words(data)


#Lemmatization

'''
In stemming, it doesn't check the meaning of the word! It simply chops off end of the word.

Lemmatization is more accurate as it uses more informed analysis to create groups of words
with similar meaning based on the context around the word.

Lemmatizers always return dictionary words. Because of the additional context considered, it is more
accurate, but costs more as well.
'''

wn= nltk.WordNetLemmatizer()


def lemmatizer_words(data):
    for ind in range(0,len(data)):
        data["text_new"].loc[ind]=[wn.lemmatize(i) for i in data["text_new"].loc[ind]]
    return data
data= lemmatizer_words(data)


# If lemmatizer will cause to have very long computational cost, then, you can opt for the simple stemmer

# When accuracy is preferred more than speed, you can use lemmatizing over the stemming

# Let's do the vectoring!

'''

Vectorizing is the process of encoding text as integers to create feature vectors

'''



'''
STEPS

1. Raw text - model cannot distinguish the words
2. Tokenize - tell the model what to look at
3. Clean text - remove stop words, punctuations, stemming, etc
4. Vectorize - convert text to numeric forms
5. Machine learning algorithm - fit/train model
6. Spam filter - system to filter email


Raw text needs to be converted to the numbers, so that thee algorithms for ML can understand
'''



# Let's do the count vectorizing first!

from sklearn.feature_extraction.text import CountVectorizer

data_new=data
data_new["text_new"]=[",".join(i) for i in data_new["text_new"]]

# count_vect= CountVectorizer()
# x= count_vect.fit_transform(data_new["text_ne"
#            "w"])
# names_= count_vect.get_feature_names()
#
# new_data= pd.DataFrame(x.toarray())
#
# new_data.columns=names_
#
# print(new_data)


# Let's do the N-grams vectorizing first!

from sklearn.feature_extraction.text import CountVectorizer
# The default value for this is 1,1

count_vect= CountVectorizer(ngram_range=(2,2))

x= count_vect.fit_transform(data_new["text_ne"
           "w"])
names_= count_vect.get_feature_names()

new_data= pd.DataFrame(x.toarray())

new_data.columns=names_

print(new_data)


# Now, you need to do the feature engineering part to train your data with the given spam and ham labels!

'''
- For creating the new features, you can;

1. Length of the text field 
2. Percentage of the characters that have punctuation in the text
3. Percentage of the characters that are capitalized

- Also, you can apply transformation on your data, such as square the data or square root! 
- If you have skewed data, you need to apply the log transformation 
- Additionally, you need to standardize your data! Scale in a certain values

'''

# Let's create the features! But, let's do this on a clean data!
# Let's hypothesize that the spam messages are tend to be longer than the text messages!


data_f_path="../SMSSpamCollection.tsv"
data_f=pd.read_table(data_f_path,header=None)
data_f.columns=["label","text"]

data_f["text len"] = data_f["text"].apply(lambda x: len(x)-x.count(" "))

new_data["text len"]= data_f["text len"]

import matplotlib.pyplot as plt
import numpy as np
bins= np.linspace(0,200,40)

plt.hist(data_f[data_f["label"]=="spam"]["text len"],bins=bins, alpha= 0.5,  label="spam")
plt.hist(data_f[data_f["label"]=="ham"]["text len"],bins=bins, alpha= 0.5,  label="ham")
plt.legend(loc="upper left")
# plt.show()

# Here, we see that spam messages are indeed longer than the ham messages, so we can use this created features!

# Now, let's look at the distribution of the created feature!
bins= np.linspace(0,200,40)
plt.hist(data_f["text len"],bins)
# plt.show()



# This is not good for the transformation, since there is no skewness observed here!

'''

Transformation -> Square each value or taking the square root of the each value.

Box-Cox Power transformation is very common one

'''


'''

How to evaluate our models?

- There are a lot of different tools;

1. Holdout test set
Sample of data not used in fitting for the purpose of evaluating the model's ability to generalize unseen data

- K-Fold Cross-Validation
The full data set is divided into k-subsets and the holdout method is repeated k times. Each time, one of the k-subsets is used as the
test set and the other k-1 subsets are put together to be used to train the model.


Example of 5-fold Cross-Validation -

Let's say you have 1000 examples of a sample, and you divide the sample to 5.

Then, you can use one of the sample group (n=200) to test the prediction, by calculating the accuracy. You can do this five times.
If the values are almost similar, then, your model works well in many conditions! You need to take the average of the accuracy at the
end.

Also, there are three metrics you can check for the testing!

1. Accuracy

It is the total correctly predicted value over the whole sample size ->


2. Precision

It is the true positives you found over the true positive and false positive. It gives you to tailor the aggressiveness of the model.

If false positives are really costly, then, you need to optimize your model according to the precision metric.


3. Recall

It is the true positives you found ovr the true positive and false negative.

'''

# new_data["label"]= data_f["label"]

# Random forest models!


from sklearn.ensemble import RandomForestClassifier

print(dir(RandomForestClassifier))

# Some important methods are feature importances, fit, predict


'''

Hyperparameters

1. Maxdepth - How deep each decision tree is 
2. n_estimators - How many decision trees will be built in your random forest

'''

from sklearn.model_selection import KFold, cross_val_score

# n_jobs=-1 allows the algorithm run faster! Because the process run in parallel

rf= RandomForestClassifier(n_jobs=-1)
k_fold= KFold(n_splits=5)

new_data=new_data[0:1000]
data_f=data_f[0:1000]

values= cross_val_score(rf,new_data,data_f["label"], cv=k_fold,scoring="accuracy",n_jobs=-1)

print("Cross Validation Scores= {}".format(values))

'''

Cross-validation is usually the preferred method because it gives your model the opportunity to train 
on multiple train-test splits. This gives you a better indication of how well your model will perform 
on unseen data. Hold-out, on the other hand, is dependent on just one train-test split. That makes the 
hold-out method score dependent on how the data is split into train and test sets.


The hold-out method is good to use when you have a very large dataset, you’re on a time crunch, or you 
are starting to build an initial model in your data science project. Keep in mind that because 
cross-validation uses multiple train-test splits, it takes more computational power and time to run than 
using the holdout method.


'''


# Let me explore the randomforest classifiers with the hold-out set

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

# Let's split the data in a way that 80 percent will be the training data set!

X_train,X_test,y_train,y_test= train_test_split(new_data,data_f["label"],test_size=0.2)

# Let me enter the hyperparameters for the classifier
rf= RandomForestClassifier(n_estimators=50, max_depth=20,n_jobs=-1)

# Let me fit the data
rf_model= rf.fit(X_train,y_train)

# Let me check the most important feature!

my_feature_imp= sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[0:10]
print(my_feature_imp)

y_pred= rf_model.predict(X_test)

precision,recall, fscore, support =score(y_test,y_pred,pos_label="spam",average="binary")

print("Precision= {}".format(round(precision,4)))

