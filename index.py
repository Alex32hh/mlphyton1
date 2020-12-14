from flask import Flask
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df = pd.read_csv('files/train.csv')
conversion_dict = {0:'Real',1:'Fake'}
df['label'] = df['label'].replace(conversion_dict)

# separa as noticias verdadeiras das falsas
dReal = df[df.label != "Real"]
dFake = df[df.label != "Fake"]

# print(df.label.value_counts())

x_train,x_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.25,random_state=7,shuffle=True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.75) 

vec_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = tfidf_vectorizer.transform(x_test.values.astype('U'))

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)

y_pred = pac.predict(vec_test)
score = accuracy_score(y_test,y_pred)
print(f'Pac Accuracy :{round(score*100,2)}%')


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
    vec_newstest = tfidf_vectorizer.transform([newtext])
    y_pred1 = pac.predict(vec_newstest)
    return y_pred1[0]


text = 'Giuliani has tested positive for the coronavirus, Trump says.Rudy Giuliani, the personal and campaign lawyer for President Trump, during an appearance before the Michigan House Oversight Committee on Wednesday.Rudy Giuliani, the personal and campaign lawyer for President Trump, during an appearance before the Michigan House Oversight Committee on Wednesday.Credit...Jeff Kowalsky/Agence France-Presse — Getty Images.Rudolph W. Giuliani, the former New York City mayor and President Trump’s personal and campaign lawyer, has tested positive for the coronavirus, Mr. Trump announced on Twitter on Sunday.“@RudyGiuliani, by far the greatest mayor in the history of NYC, and who has been working tirelessly exposing the most corrupt election (by far!) in the history of the USA, has tested positive for the China Virus. Get better soon Rudy, we will carry on!!!” Mr. Trump wrote. It was unclear why Mr. Trump was the one announcing it.Mr. Giuliani was at Georgetown University Medical Center, according to a person who was aware of his condition but not authorized to speak publicly. Mr. Giuliani, at age 76, is in the high-risk category for the virus.Mr. Giuliani has repeatedly been exposed to the virus through contact with infected people, including during Mr. Trump’s preparation for his first debate against President-elect Joseph R. Biden Jr. in September, just before the president tested positive, as well as when he appeared with his son, Andrew, at a news conference at the Republican National Committee headquarters about two weeks ago. Andrew Giuliani, who works as an aide in the White House, said on Nov. 20 that he had tested positive, days after Donald Trump Jr. did.  Mr. Giuliani has been acting as the lead lawyer for Mr. Trump’s efforts to overthrow the results of the election. He has repeatedly claimed he has evidence of widespread fraud, but he has declined to submit that evidence in legal cases he has filed.His infection is the latest in a string of outbreaks among those in the president’s orbit. Boris Epshteyn, a member of the Trump campaign legal team, tested positive late last month. The same day, Mr. Giuliani attended a meeting of Republican state lawmakers in Pennsylvania about allegations of voting irregularities. One of the lawmakers at that meeting was notified shortly after while at the White House that he had tested positive.Mark Meadows, the president’s chief of staff, and at least eight others in the White House and Mr. Trump’s circle, tested positive in the days before and after Election Day.Mr. Trump was hospitalized on Oct. 2 after contracting the coronavirus. Kayleigh McEnany, the president’s press secretary, Corey Lewandowski, a campaign adviser, and Ben Carson, the housing secretary, are among those in the president’s circle who have tested positive this fall.Mr. Giuliani appeared on Fox News earlier on Sunday. Speaking with the host Maria Bartiromo via satellite, Mr. Giuliani repeated baseless claims about fraud in Georgia and Wisconsin on “Sunday Morning Futures.” When asked if he believed Mr. Trump still had a path to victory, he said, “We do.”Tracking Coronavirus Infections in the White House and Trump’s Inner Circle After a small White House outbreak in September and a wave of cases in early October, the coronavirus has returned to the top of the Trump administration.Residents of nursing homes could receive vaccine by the end of December, official says.A nursing home in Redmond, Wash., in October. Widespread vaccination of the elderly could begin in days, an administration official said.A nursing home in Redmond, Wash., in October. Widespread vaccination of the elderly could begin in days, an administration official said.Credit...Grant Hindsley for The New York Times.Trump administration officials on Sunday laid out an ambitious timetable for the rollout of the first coronavirus vaccine in the United States, rebuking President-elect Joseph R. Biden Jr.’s criticism that there was “no detailed plan that we’ve seen” for getting people immunized.Dr. Moncef Slaoui, chief science adviser of Operation Warp Speed, the administration’s program to develop and deploy vaccines, said that residents of long-term care facilities will receive the first round of vaccinations by mid-January, perhaps even by the end of December. In some states, this group accounts for about 40 percent of deaths from the coronavirus.The timing assumes that the Food and Drug Administration authorizes the vaccine, made by Pfizer, this week or shortly thereafter. An advisory committee to the agency will meet on Thursday to review the data on safety and efficacy.If the agency authorizes the vaccine, distribution could begin as soon as the end of this week, Dr. Slaoui added. “By end of the month of January, we should already see quite a significant decrease in mortality in the elderly population,” he said on CNN’s “State of the Union.”Barring unexpected problems with manufacturing the vaccine, most Americans at high risk from coronavirus infection should be vaccinated by mid-March, and the rest of the population by May or June, he added.President-elect Biden sounded a considerably more skeptical note on Friday, saying that there was “no detailed plan that we’ve seen, anyway, as to how you get the vaccine out of a container, into an injection syringe, into somebody’s arm.”Dr. Slaoui said his team expected to meet Mr. Biden’s advisers this week and brief them on details of the plan for the vaccines’ distribution.Britain has already approved the Pfizer vaccine and expects to begin immunizing its population this week. Like the F.D.A., European regulators are still examining data on the vaccine’s safety and effectiveness.A second vaccine, made by Moderna, also has been submitted to the F.D.A. for emergency authorization.  Dr. Slaoui was optimistic about long-term protection from the vaccine. The elderly or people with compromised immune systems might need a booster in three to five years, he said, but for most people the vaccine should remain effective for “many, many years.”Still, it’s unclear whether those who have been immunized may still spread the virus to others. “The answer to that very important question” should be known by mid-February, he said.Up to 15 percent of those receiving the shots experience “significant, not overwhelming” pain at the injection site, which usually disappears in a day or two, Dr. Slaoui told CBS’s “Face the Nation,” also on Sunday.Vaccines have not yet been tested in children under 12, but Dr. Slaoui said that clinical trials in adolescents and toddlers might produce results by next fall.Operation Warp Speed was expected to have 100 million doses of the Pfizer vaccine by December, a number that has since been slashed by more than half.Although the clinical trials were completed faster than expected because of the high level of virus transmission in the United States, manufacturing problems scaled down the expected number of available doses to 40 million.Dr. Slaoui warned of possible further delays. “This is not an engineering problem. These are biological problems, they’re extremely complex,” he said. “There will be small glitches.”Christmas tree sales are booming as pandemic-weary Americans seek solace.'

# translator = Translator()
# result = translator.translate('Mitä sinä teet')
# print(result.text)

print(findlabel(text))

# calcula o score de cada acerto positivo 
if(findlabel(text) == 'Real'):
    result = int(round((sum([1 if findlabel(text)=="Real" else 0 for i in range(len(text))])/df_true["text"].size)*100,2))
    print(f'Real Assurance% :{result}%')
else:
    result = int(round((sum([1 if findlabel(text)=="Fake" else 0 for i in range(len(text))])/df_fake["text"].size)*100,2))
    print(f'Fake Assurance% :{result}%')
 
