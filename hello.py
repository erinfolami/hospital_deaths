import pandas as pd
from  matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost


df = pd.read_csv("data.csv")
# Shape of original dataframe = (91713, 244)

# drop null values
df = df.dropna(axis=0)
#"hospital_death","patient_id","encounter_id"
x = df.drop(["hospital_death","patient_id","hospital_id"],axis=1)
y = df["hospital_death"]

# Using 20% of the data for testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

model = xgboost.XGBClassifier()
#subsample=0.9  colsample_bylevel=0.5  colsample_bytree=0.5  min_child_weight=3
model = model.fit(x_train,y_train)
model = model.score(x_test,y_test)
print(model)

# model = xgboost.XGBClassifier()
# #original score = 0.6363636363636364
#
# num = []
# for i in range(0,700):
#     num.append(i)
#
# param = {"subsample":[0.2,0.1,0.3}
# #,"min_child_weight":num,"learning_rate":[0.001,0.1,0.01,0.5,0.8],"n_estimators":num
# search = GridSearchCV(model,param,cv=10,scoring="accuracy")
# search = search.fit(x_train,y_train)
# print(search.best_params_)
# #{'colsample_bylevel': 0.7, 'colsample_bytree': 0.8, 'subsam//+ple': 0.3}