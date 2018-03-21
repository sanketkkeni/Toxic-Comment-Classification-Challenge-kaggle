# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:25:59 2018

@author: Sanket Keni
"""
import pickle

train_real = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\train.csv').fillna(' ')
test_real = pd.read_csv('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\test.csv').fillna(' ')

##################
q=word_vectorizer.get_feature_names()
idf = word_vectorizer.idf_
dict1= (dict(zip(word_vectorizer.get_feature_names(), idf)))
#word_vectorizer.vocabulary_
#word_vectorizer.stop_words_ 
l1=[]
l2=[]
for key in dict1:
    l1.append(key)
    l2.append(dict1[key])
df1 = pd.concat([pd.DataFrame(l1),pd.DataFrame(l2)],axis =1)
df1.columns = ["key","value"]
df1 = df1.sort_values(by = "value", ascending = False)
##############################


test_index_bad_data_list.append(82288)
test_index_bad_data_list = list(set(test_index_bad_data_list))

22852 36593
train_index_bad_data_list.append(57594)
train_index_bad_data_list = list(set(train_index_bad_data_list))

df1.head(20)
mask = np.column_stack([train['comment_text'].str.contains(r"teabag", na=False)])
train_real.loc[mask.any(axis=1)]["comment_text"].head()
mask = np.column_stack([test['comment_text'].str.contains(r"titoxd", na=False)])
test_real.loc[mask.any(axis=1)]["comment_text"].head()



train.iloc[80030][1].count("yourselfgo")

list_outlier_words = list(df1["key"][:100])

for i in list_outlier_words:    
    mask = np.column_stack([train['comment_text'].str.contains(i, na=False)])
    get_index = list(train_real.loc[mask.any(axis=1)]["comment_text"].index)
    for j in get_index:
        count = 0
        text = train_real.iloc[j][1]
        if (text.count(i)>30): # threshold value
            count += 1
            print ("fount outliers", count)
            word = " "+i+" "
            text=re.sub(re.escape(word)," ",text) + " " +i
            train_real.iat[j,1] = text



list(train_real.loc[mask.any(axis=1)]["comment_text"].index)






# make changes to train and test
'''
comment=re.sub("\s+"," ",comment)



22998
train.iloc[137604]
submission.iloc[152552]

'''



feat_list = list(df1["key"])
feat_index = list(df1.index)

# taking which index has bad data
test_index_with_bad_data = {}
train_index_with_bad_data = {}
for index,i in enumerate(feat_list[:100]):
    print(index)
    # test data
    temp=[]
    mask_test = np.column_stack([test['comment_text'].str.contains(re.escape(i), na=False)])
    for j in range(len(mask_test)):
        if(mask_test[j][0] == True):
            temp.append(j)
    test_index_with_bad_data[index] = temp
    
    # train data
    temp =[]
    mask_train = np.column_stack([train['comment_text'].str.contains(re.escape(i), na=False)])
    for j in range(len(mask_train)):
        if(mask_train[j][0] == True):
            temp.append(j)
    train_index_with_bad_data[index] = temp



ind_val_test = []
ind_val_train = []
for i in range(len(test_index_with_bad_data)):
    l1 = test_index_with_bad_data[i]
    l2 = train_index_with_bad_data[i]
    if((len(l1)+len(l2)) == 1):
        if (len(l1) == 1):
            
            ind_val_test.append(l1[0])
        else:
            ind_val_train.append(l2[0])
           
    










'''
df1.iloc[55]

test["comment_text"][30849]
feat_index[57]
test.iat[30849,1] = "how do u do?"




l = str(test_real.iloc[30849]["comment_text"])
'''

def unique_text(l):
    return ' '.join(set(l.split()))


for i in range(100):
    if (len(test_index_with_bad_data[i]) == 1):        
        index = test_index_with_bad_data[i][0]
        print(index)
        text = test_real.iloc[index][1]
        test_real.iat[index,1] = unique_text(text)
        
    if (len(train_index_with_bad_data[i]) == 1):        
        index = train_index_with_bad_data[i][0]
        print(index)
        text = train_real.iloc[index][1]
        #train["comment_text"][train_index_with_bad_data[i]] = unique_text(text)
        train_real.iat[index,1] = unique_text(text)


test_index_bad_data_list =[]
train_index_bad_data_list =[]
for i in range(100):
    if (len(test_index_with_bad_data[i]) == 1):        
        index = test_index_with_bad_data[i][0]
        test_index_bad_data_list.append(index)
        
    if (len(train_index_with_bad_data[i]) == 1):        
        index = train_index_with_bad_data[i][0]
        train_index_bad_data_list.append(index)


with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_test_index.pkl', 'wb') as f:
    pickle.dump(test_index_bad_data_list, f)
with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_train_index.pkl', 'wb') as f:
    pickle.dump(train_index_bad_data_list, f)


with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_test_index.pkl', 'rb') as f:
    test_index_bad_data_list = pickle.load(f)
with open('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\toxic comment\\bad_train_index.pkl', 'rb') as f:
    train_index_bad_data_list = pickle.load(f)

for i in train_index_bad_data_list:
    text = train_real.iloc[i][1]
    train_real.iat[i,1] = unique_text(text)
        



'''
train["comment_text"][6201]


train_real.iloc[104265][1]
test_real.iloc[56313][1]
'''


for i in test_index_bad_data_list:
    print(test_real.iloc[i][1])

 train_real.iloc[152900][1]
9



for i in ind_val_test:
    text = test.iloc[i][1]
    test.iat[i,1] = unique_text(text)
        
 '''       
    if (len(train_index_with_bad_data[i]) == 1):        
        index = train_index_with_bad_data[i][0]
        print(index)
        text = train_real.iloc[index][1]
        #train["comment_text"][train_index_with_bad_data[i]] = unique_text(text)
        train_real.iat[index,1] = unique_text(text)
'''









def print2(i):
    print()
    print(i)
    print("Before--------------------------------",test_real.iloc[i][1])
    print()
    print("After---------------------------------",test.iloc[i][1])


for i in range(4):
    print2(ind_val_test[i+30])
    
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

test_index_bad_data_list = ind_val_test

ind_val_test.remove(111567)

print2(106390)



[6828,
 141503,
 121367,
 90258,
 106975,
 106390,
 87823,
 11958,
 20266,
 59934,
 84591,
 71360,
 47590,
 44209,
 84881,
 125057,
 135550,
 56506,
 128815,
 69007,
 119574,
 29834,
 64406,
 110193,
 16567,
 110193,
 3545,
 110022,
 46963,
 87334,
 126241,
 64406,
 78755,
 137917,
 90365,
 100793,
 103144,
 111567,
 115381,
 82433,
 28733,
 101536,
 75035]








[56313,
 128782,
 40223,
 53408,
 104994,
 91351,
 94644,
 137604,
 70544,
 6201,
 57173,
 35028,
 151690,
 97397,
 6572,
 152900]

train.iat[6201,1] = "HOMELAND SECURITY FUCK"






