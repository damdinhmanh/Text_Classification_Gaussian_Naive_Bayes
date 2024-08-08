import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

##-------------------------METHOD----------------------------------------
def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    translator = str.maketrans('','', string.punctuation)
    return text.translate(translator)

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    return [tk for tk in tokens if tk not in stop_words]

def stemming(tokens):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(tk) for tk in tokens]

def process_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens

def create_dictionary(messages):
    dictionary = []

    for tokens in messages:
        for tk in tokens:
            if tk not in dictionary:
                dictionary.append(tk)
    return dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))

    for tk in tokens:
        if tk in dictionary:
            features[dictionary.index(tk)] += 1
    return features

def predict_text(text, model, dictionary):
    processed_text = process_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls

##-------------------------END METHOD--------------------------------------

#for debug dataframe
# df = pd.read_csv("D:\AI_ML_DL\Deep_Learning_Project\Spam_Text_Classification_Naive_Bayes\cls_spam_text_cls.csv")

df = pd.read_csv("cls_spam_text_cls.csv")
# print(df.shape)

messages = df['Message']
labels = df['Category']

le = LabelEncoder()
y = le.fit_transform(labels)

messages = [process_text(msg) for msg in messages]

dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=0)

model = GaussianNB()
print('Start training')
model = model.fit(x_train, y_train)
print('Training completed!')

y_val_predict = model.predict(x_val)
y_test_predict = model.predict(x_test)

val_accuracy = accuracy_score(y_val, y_val_predict)
test_accuracy = accuracy_score(y_test, y_test_predict)

print('Val accuracy: {}'.format(val_accuracy))
print('Test accuracy: {}'.format(test_accuracy))

test_input = ' Your 455555 Account for 3434 shows 43434 unredeemed Bonus'
prediction_cls = predict_text(test_input, model, dictionary)

print('Predict class: {}'.format(prediction_cls))
