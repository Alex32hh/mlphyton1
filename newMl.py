from flask import Flask,request,jsonify,render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import pickle


df = pd.read_csv('files/train.csv')
conversion_dict = {0:'Real',1:'Fake'}
df['label'] = df['label'].replace(conversion_dict)

# separa as noticias verdadeiras das falsas
dReal = df[df.label != "Real"]
dFake = df[df.label != "Fake"]

# print(df.label.value_counts())

x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.25,random_state=7,shuffle=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.75)
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pickle", "wb"))


vectorizer = joblib.load('tfidf_vectorizer.pickle')
vec_train = vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = vectorizer.transform(x_test.values.astype('U'))


pac = PassiveAggressiveClassifier(max_iter=50)
pickle.dump(pac, open("pac.pickle", "wb"))

vectorizerPac = joblib.load('pac.pickle')
vectorizerPac.fit(vec_train,y_train)

y_pred = vectorizerPac.predict(vec_test)
score = accuracy_score(y_test,y_pred)
# print(f'Pac Accuracy :{round(score*100,2)}%')


df_true=pd.read_csv('files/True.csv')
df_true['label']='Real'
df_true_rep=[df_true['text'][i].replace('WASHINGTON (Reuters) - ','').replace('LONDON (Reuters) - ','').replace('(Reuters) - ','') for i in range(len(df_true['text']))]
df_true['text']=df_true_rep
df_fake=pd.read_csv('files/Fake.csv')
df_fake['label']='Fake'
df_final=pd.concat([df_true,df_fake])
df_final=df_final.drop(['subject','date'], axis=1)

# df_fake

def findlabel(newtext):
    vec_newstest = vectorizer.transform([newtext])
    y_pred1 = vectorizerPac.predict(vec_newstest)
    return y_pred1[0]


text = "Giuliani has tested positive for the coronavirus, Trump says.Rudy Giuliani, the personal and campaign lawyer for President Trump, during an appearance before the Michigan House Oversight Committee on Wednesday.Rudy Giuliani, the personal and campaign lawyer for President Trump, during an appearance before the Michigan House Oversight Committee on Wednesday.Credit"


print(findlabel(text))

# calcula o score de cada acerto positivo 
if(findlabel(text) == 'Real'):
    result = int(round((sum([1 if findlabel(text)=="Real" else 0 for i in range(len(text))])/df_true["text"].size)*100,2))
    # print(f'Real Assurance% :{result}%')
else:
    result = int(round((sum([1 if findlabel(text)=="Fake" else 0 for i in range(len(text))])/df_fake["text"].size)*100,2))
    # print(f'Fake Assurance% :{result}%')
 

