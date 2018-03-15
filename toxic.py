# Sanket Keni 
'''
CV score for class toxic is 0.9682470069323074
CV score for class severe_toxic is 0.9872657124996594
CV score for class obscene is 0.9843951465629428
CV score for class threat is 0.9810181759466508
CV score for class insult is 0.9767931386642288
CV score for class identity_hate is 0.9730722219508444
Total CV score is 0.9784652337594388

Kaggle - 0.9738
'''
import numpy as np
import pandas as pd
from playsound import playsound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import timeit

# notify when code has completed execution
def audio():
    playsound('C:\\Users\\Sanket Keni\\Music\\gang.mp3')


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\train.csv').fillna(' ')
test = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])



word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=5000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
#audio()

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 3),
    max_features=5000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
audio()

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')
    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\out.csv', index=False)
audio()
