### INSTALL LIBRARIES ###
import re
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import gensim
from sklearn.model_selection import train_test_split
import spacy
import pickle
import warnings
#warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
print('Libraries Done')

### DATA EXPLORATION ###
train=pd.read_csv("/home/cle-157/Sara/sentiment_analysis/Dataset/train.csv")
#print(train.head(10))
#print(len(train))
#print(train['sentiment'].unique())
#print(train.groupby("sentiment").nunique())
#Let's keep only the columns that we're going to use
train=train[["selected_text","sentiment"]]
#print(train.head(10))
#print(train['selected_text'].isnull().sum())
train["selected_text"].fillna("No Content",inplace=True)

### DATA CLEANING ###
def depure_data(data):
    # Removing URLs
    url_pattern=re.compile(r'https?://\S+|www\.\S+')
    data=url_pattern.sub(r'', data)
    # Removing emails
    data=re.sub('\S*@\S*\s?', '',data)
    #Remove new line characters
    data=re.sub('\s+', ' ',data)
    # Remove distracting single quotes
    data=re.sub("\'", "",data)
    return data

#Splitting pd.Series to list
temp=[]
data_to_list=train["selected_text"].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
#print(list(temp[:5]))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

data_words=list(sent_to_words(temp))
#print(data_words[:5])

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data=[]
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
#print(data[:5])
data=np.array(data)

### LABEL ENCODING ###
labels=np.array(train["sentiment"])
y=[]
for i in range(len(labels)):
    if labels[i]=="neutral":
        y.append(0)
    if labels[i]=="negative":
        y.append(1)
    if labels[i]=="positive":
        y.append(2)
y=np.array(y)
labels=tf.keras.utils.to_categorical(y,3,dtype="float32")
del y

### DATA SEQUENCING ###
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

max_words=5000
max_len=200

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences=tokenizer.texts_to_sequences(data)
tweets=pad_sequences(sequences,maxlen=max_len)
#print(tweets)

#Splitting the data
X_train,X_test,y_train,y_test=train_test_split(tweets,labels,random_state=0)
#print(len(X_train),len(X_test),len(y_train),len(y_test))

### MODEL BUILDING ###


### Bidirectional LTSM model ###
model2=Sequential()
model2.add(layers.Embedding(max_words,40,input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20,dropout=0.6)))
model2.add(layers.Dense(3,activation="softmax"))
model2.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=['accuracy'])
#checkpoint2=ModelCheckpoint("/home/cle-157/Sara/sentiment_analysis/best_model22.hdf5",monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
#history=model2.fit(X_train,y_train,epochs=3,validation_data=(X_test,y_test),callbacks=[checkpoint2])
model2.fit(X_train,y_train,epochs=3,validation_data=(X_test,y_test))
#model2.save("/home/cle-157/Sara/sentiment_analysis/model.h5")
model2.save("my_model")


### Best model validation ###
# best_model=keras.models.load_model("/home/cle-157/Sara/sentiment_analysis/best_model22.hdf5")
# test_loss,test_acc=best_model.evaluate(X_test,y_test,verbose=2)
# print("Model accuracy",test_acc)
#
# predictions=best_model.predict(X_test)


### Confusion matrix ###
# from sklearn.metrics import confusion_matrix
# matrix=confusion_matrix(y_test.argmax(axis=1),np.around(predictions,decimals=0).argmax(axis=1))
#print(matrix)

# import seaborn as sns
# conf_matrix=pd.DataFrame(matrix,index=['Neutral','Negative','Positive'],columns=['Neutral','Negative','Positive'])
# #Normalizing
# #conf_matrix=conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:,np.newaxis]
# conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(15,15))
# sns.heatmap(conf_matrix,annot=True,annot_kws={"size": 15})





# sentiment=["Neutral","Negative","Positive"]
# sequence=tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])
# test=pad_sequences(sequence,maxlen=max_len)
# a=sentiment[np.around(best_model.predict(test),decimals=0).argmax(axis=1)[0]]
# print(a)

