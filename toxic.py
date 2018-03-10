import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


test = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\test.csv")
train = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\train.csv")
sample = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\sample_submission.csv")


def output_one_classifier(test_col):
    
    df_x=train["comment_text"]
    df_y=train[test_col]
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

    cv = TfidfVectorizer(min_df=1,stop_words='english', max_features=1000)
    x_train_trans=cv.fit_transform(x_train)
    a=x_train_trans.toarray()

    x_test_trans=cv.transform(x_test)
    x_test_trans.toarray()

    mnb = MultinomialNB()
    mnb.fit(x_train_trans,y_train)
    predictions=mnb.predict(x_test_trans)
    a=np.array(y_test)
    count=0
    for i in range (len(predictions)):
        if predictions[i]==a[i]:
            count=count+1
    accuracy = count*100/len(predictions)
    print("cross validation accuracy for", test_col,"column is:", accuracy)     
    df_test=test["comment_text"]    
    test_inp = cv.transform(df_test)
    test_inp.toarray()    
    out = mnb.predict(test_inp)
    return out
    
toxic = output_one_classifier("toxic")  
severe_toxic = output_one_classifier("severe_toxic")   
obscene = output_one_classifier("obscene")      
threat = output_one_classifier("threat")      
insult = output_one_classifier("insult")     
identity_hate = output_one_classifier("identity_hate")     
id = test["id"].values
    
out_df = pd.DataFrame({"id": id, "toxic": toxic, "severe_toxic": severe_toxic, "obscene": obscene, "threat": threat, "insult": insult, "identity_hate": identity_hate})   
    
out_df.to_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\out.csv", index=False)    
    






