import pandas as pd
from string import punctuation
import emoji
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib as jb
from xgboost import XGBClassifier

df1 = pd.read_csv('d1.csv')
df2 = pd.read_csv('d2.csv')
df3 = pd.read_csv('d3.csv')
df4 = pd.read_csv('d4.csv')
df5 = pd.read_csv('d5.csv')

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

labels = ['COMMENT_ID', 'AUTHOR', 'DATE'] 
df.drop(labels, axis=1, inplace=True)

labels = labels = set([*punctuation, '﻿', *emoji.EMOJI_DATA.keys()])

def cleantext(text):
    text = str(text)
    return ''.join([x for x in text if x not in labels])

df['CONTENT'] = df['CONTENT'].apply(cleantext)

X = df['CONTENT']
y = df['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print("Total texts: ", df["CONTENT"].count())
real = df[df["CLASS"] == 0.0].shape[0]
fake = df[df["CLASS"] == 1.0].shape[0]
print("Real Comments: ", real)
print("Fake Comments: ", fake)

print("X_train: ", len(X_train))
print("y_train: ", len(y_train))

pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),  
    ('tfidf_transformer', TfidfTransformer()),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5))
])

model = pipeline.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Ham', 'Spam'])
plt.show()


jb.dump(model, "SpamPredicter.pkl")
