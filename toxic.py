# Sanket Keni 
'''
CV score for class toxic is 0.9758815956729977
CV score for class severe_toxic is 0.9885067270242905
CV score for class obscene is 0.9919493883065732
CV score for class threat is 0.9866684407022007
CV score for class insult is 0.9806593278329583
CV score for class identity_hate is 0.981040742648163
Total CV score is 0.9841177036978639
Public LB - 0.9787
Final LB - 0.9793
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
import pickle
from slackclient import SlackClient
stemmer = nltk.stem.snowball.SnowballStemmer('english')

############### send notification on smartphone

def slack_message(message, channel):
    token = '*************'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, 
                text=message, username='My Sweet Bot',
                icon_emoji=':robot_face:')

slack_message("Execution complete", "U9U1CAH2S")



# notify when code has completed execution
def audio():
    playsound('C:\\Users\\Sanket Keni\\Music\\notification.mp3')

def unique_text(l):
    return ' '.join(set(l.split()))

train = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\train.csv').fillna(' ')
test = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\test.csv').fillna(' ')
'''
train['comment_text']=train['comment_text'].apply(lambda x :clean(x))
test['comment_text']=test['comment_text'].apply(lambda x :clean(x))
'''
################ Data Cleaning
train.iat[6201,1] = "HOMELAND SECURITY FUCK"
test.iat[59934,1] = "fuck you mather serbia fuck"    
test.iat[87823,1] = " ROFL SPAM "
test.iat[121367,1] = "YOU ARE A GODDAMED FAGGOT"
test.iat[47590,1] = "nguyen do is a cock sucker"
test.iat[6828,1] = " fuck mother fucker "
test.iat[125057,1] = " DONKEY SEX "
test.iat[126241,1] = " penis "
test.iat[110022,1] = " PETER SYMOND IS HOMO "
test.iat[111567,1] = " BONER ORGASM "
test.iat[28733,1] = " PENIS IN VAGINA "
test.iat[82433,1] = " BITCH bot "
test.iat[5359,1] = " anal "
test.iat[16575,1] = " blaa "
test.iat[55276,1] = " asshole fuck "
test.iat[61389,1] = " nigger "
train.iat[69439,1] = " From the rest of the world \n go fuck yourself "
test.iat[12529,1] = " you mad "
test.iat[78325,1] = " nigger jew bastard "
test.iat[144015,1] = " FAGGOT nigger jew "
train.iat[35316,1] = " Marxism LONG LIVE ANONYMOUS PHILIPPINES "
train.iat[67038,1] = " Anthony Bradbury sucks cocks and eats shit "
train.iat[100924,1] = " Anthony Bradbury sucks cocks and eats shit "
train.iat[141015,1] = " Anthony Bradbury sucks cocks and eats shit "
train.iat[121718,1] = " BUTTSECKS butt sex "
train.iat[41152,1] = " SUPRISE BUTTSECKS butt sex "
train.iat[4584,1] = " Dick You. You boy. Terisoaio! Yeah Yeah Yeah "



with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_train_index.pkl', 'rb') as f:
    train_index_bad_data_list = pickle.load(f)
with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_test_index.pkl', 'rb') as f:
    test_index_bad_data_list = pickle.load(f)
## trywith *10
for i in train_index_bad_data_list:
    text = train.iloc[i][1]
    train.iat[i,1] = unique_text(text)
    
for i in test_index_bad_data_list:
    text = test.iloc[i][1]
    test.iat[i,1] = unique_text(text)
    
train.iat[152900,1] = " liz jone hairi old cunt " + train.iloc[152900][1]        
##################


def cleaned(comment):
    comment=comment.lower()
    comment=re.sub("\\n"," ",comment)
    comment=re.sub("\d{1,}"," ",comment)
    comment=re.sub("\.{1,}",".",comment)
    comment=re.sub("\:{1,}"," ",comment)
    comment=re.sub("\;|\=|\%|\^|\_"," ",comment)
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
    comment=re.sub(r'[^\x00-\x7F]+',' ', comment) # remove non ascii
    comment=re.sub('(u{2,})', 'u', comment) 
    comment=re.sub('(f{2,})', 'f', comment) 
    comment=re.sub('(k{2,})', 'k', comment) 
    comment=re.sub('(o{3,})', 'o', comment) 
    comment=re.sub('(c{2,})', 'c', comment) 
    comment=re.sub('(y{2,})', 'y', comment) 
    comment=re.sub(r' cawk ', ' cock ', comment) 
    comment=re.sub(r' phck ', ' fuck ', comment) 
    comment=re.sub(r'mothjer', 'mother', comment) 
    comment=re.sub("\s+"," ",comment)
    comment = ' '.join( [w for w in comment.split() if len(w)>1])
    comment = ' '.join( [stemmer.stem(w) for w in comment.split()])
    comment = comment.strip()
    return comment


train['comment_text']=train['comment_text'].apply(lambda x :cleaned(x))
test['comment_text']=test['comment_text'].apply(lambda x :cleaned(x))
audio()

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
    max_features=14000)
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
    max_features=4000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
audio()




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
        train_target = train[class_name]
        classifier = Ridge(alpha=20, solver='auto',max_iter=100, random_state=22,  tol=0.0005)
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        classifier.fit(train_features, train_target)
        predicted = classifier.predict(test_features)        
        for index,i in enumerate(predicted):
            if(i>1):
                predicted[index] = 1
            elif(i<0):
                predicted[index] = 0
        submission[class_name] = predicted

print('Total CV score is {}'.format(np.mean(scores)))
audio()
#notify()
submission.to_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\out1.csv', index=False)
