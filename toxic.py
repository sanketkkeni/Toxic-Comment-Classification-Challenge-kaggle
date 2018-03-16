# Sanket Keni 
'''
CV score for class toxic is 0.97531893271232
CV score for class severe_toxic is 0.9882911811070457
OBSCENE class
CV score for class obscene is 0.9918359359724859
CV score for class threat is 0.9871264302579318
CV score for class insult is 0.9805883409571478
CV score for class identity_hate is 0.9810203713351426
Total CV score is 0.984030198723679
Public LB - 0.9778
'''
import numpy as np
import pandas as pd
import nltk
from playsound import playsound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import timeit
import re
############### send notification on smartphone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
url = 'https://www.pushsafer.com/api' # Set destination URL here
post_fields = {                       # Set POST fields here
	"t" : "Python code execution complete",
	"m" : "task finished",
	"d" : "a",
	"u" : url,
	"k" : "***********"
	}

#stemmer = nltk.stem.snowball.SnowballStemmer('english')

# notify when code has completed execution
def audio():
    playsound('C:\\Users\\Sanket Keni\\Music\\gang.mp3')
'''
def clean(comment):
    comment=comment.lower()
    comment=re.sub("\\n"," ",comment)
    comment=re.sub("\d{1,}","",comment)
    comment=re.sub("\(.*?\)","",comment)
    comment=re.sub("\"|:|@|,|\/|\=|;|\.|\'|\?|\!|\||\+|\~|\-|\#"," ",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\[.*?\]","",comment)
    comment=re.sub("www\S+","",comment)
    comment=re.sub("\_"," ",comment)
    comment=re.sub("http","",comment)
    comment=re.sub("\s+"," ",comment)
    return comment

def stem_and_stopword_filter(text, filter_list):    
    return [stemmer.stem(word) for word in text.split() if word not in filter_list and len(word) > 2]

stopwords = nltk.corpus.stopwords.words('english') + ['jmabel','chatspy','jpg','wikipedia','www']

def stemmed(text):
    words = stem_and_stopword_filter(text, stopwords)
    clean_text=" ".join(words)
    return clean_text    '''

train = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\train.csv').fillna(' ')
test = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\test.csv').fillna(' ')
'''
train['comment_text']=train['comment_text'].apply(lambda x :clean(x))
test['comment_text']=test['comment_text'].apply(lambda x :clean(x))

'''



def cleaned(comment):
    comment=comment.lower()
    comment=re.sub("\\n"," ",comment)
    comment=re.sub("\d{1,}","",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\:{1,}","",comment)
    comment=re.sub("\;|\=|\%|\^"," ",comment)
    comment=re.sub("\""," ",comment)
    comment=re.sub("\'{2,}","",comment)
    comment=re.sub("\/|\!"," ",comment)
    comment=re.sub("\?"," ",comment)
    comment=re.sub("\#"," ",comment)
    comment=re.sub("\,|\@|\|"," ",comment)
    comment=re.sub("\(|\)"," ",comment)
    comment=re.sub("\S+jpg"," ",comment)
    comment=re.sub("\S*wikip\S+","",comment)               
    comment=re.sub("\[.*?\]"," ",comment)
    comment=re.sub("\-"," ",comment)
    '''comment=re.sub("\"|:|@|,|\/|\=|;|\.|\'|\?|\!|\||\+|\~|\-|\#"," ",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\[.*?\]","",comment)
    comment=re.sub("www\S+","",comment)
    comment=re.sub("\_"," ",comment)
    comment=re.sub("http","",comment)'''
    comment=re.sub("\s+"," ",comment)
    comment = ' '.join( [w for w in comment.split() if len(w)>1] )
    comment = comment.strip()
    return comment

'''
l = list(train['comment_text'])

l[:50]
s = train['comment_text'][159160]
re.sub("\/"," ",s)

d = "t kwh m^ peaking. "
re.sub("\;|\=|\%|\^"," ",d)
re.sub("http","",s)
'''
'''
s = ' '.join( [w for w in s.split() if len(w)>1] )

cleaned(s)
df=train["comment_text"].apply(lambda x: len(re.findall("\[.*?\]",str(x)))+1)

'''

train['comment_text']=train['comment_text'].apply(lambda x :cleaned(x))
test['comment_text']=test['comment_text'].apply(lambda x :cleaned(x))


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


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
notify()

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:
    if (class_name in ['toxic']):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.63, solver='sag', class_weight = "balanced") # sag arge datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    
    elif(class_name in ["severe_toxic", "insult"]):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.38, solver='sag') # sag large datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
        
    elif(class_name in ["threat", "identity_hate"]):
        train_target = train[class_name]
        classifier = LogisticRegression(C=0.45, solver='sag') # sag large datasets and bivariate
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
        
    elif(class_name == "obscene"):
        print("OBSCENE class")
        train_target = train[class_name]
        classifier = Ridge(alpha=20, solver='auto',max_iter=100, random_state=22,  tol=0.0005)
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict(test_features)

print('Total CV score is {}'.format(np.mean(scores)))
notify()
submission.to_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\out.csv', index=False)
audio()
