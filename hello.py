import pandas as pd
from  matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn import preprocessing as pr
from sklearn.feature_selection import RFE
from skopt import BayesSearchCV

df = pd.read_csv("training_v2.csv")
df = df.drop(["albumin_apache","bilirubin_apache","fio2_apache","paco2_apache","pao2_apache","urineoutput_apache","d1_diasbp_invasive_max","d1_diasbp_invasive_min","d1_mbp_invasive_max","d1_mbp_invasive_min","d1_sysbp_invasive_max","d1_sysbp_invasive_min","h1_diasbp_invasive_max","h1_diasbp_invasive_min","h1_mbp_invasive_max","h1_mbp_invasive_min","d1_albumin_max","d1_albumin_min","d1_bilirubin_max","d1_bilirubin_min","d1_inr_max","d1_inr_min","d1_lactate_max","h1_albumin_max","h1_albumin_min","h1_bilirubin_max","h1_bilirubin_min"],axis=1)
df = df.drop(["h1_sysbp_invasive_max","h1_sysbp_invasive_min","d1_lactate_min","h1_bun_max","h1_bun_min","h1_calcium_max","h1_calcium_min","h1_creatinine_max","h1_creatinine_min","h1_glucose_max","h1_glucose_min","h1_hco3_max","h1_hco3_min","h1_hemaglobin_max","h1_hemaglobin_min"],axis=1)
df = df.drop(["h1_hematocrit_max","h1_hematocrit_min","h1_lactate_max","h1_lactate_min","h1_platelets_max","h1_platelets_min","h1_potassium_max","h1_potassium_min","h1_sodium_max","h1_sodium_min","h1_wbc_max","h1_wbc_min","d1_arterial_pco2_max","d1_arterial_pco2_min","d1_arterial_ph_min","d1_arterial_po2_max","d1_arterial_po2_min"],axis=1)
df = df.drop(["encounter_id","patient_id","hospital_id","d1_pao2fio2ratio_max","d1_pao2fio2ratio_min","h1_arterial_pco2_max","h1_arterial_pco2_min","h1_arterial_ph_max","h1_arterial_ph_min","h1_arterial_po2_max","h1_arterial_po2_min","h1_pao2fio2ratio_max","h1_pao2fio2ratio_min"],axis=1)
df = df.drop(["ph_apache","ventilated_apache","d1_diasbp_noninvasive_max","d1_calcium_max"],axis=1)
df = df.dropna(axis=0)

x = df.drop(["hospital_death"],axis=1)
pre  = pr.LabelEncoder()
x["ethnicity"] = pre.fit_transform(x["ethnicity"])
x["gender"] = pre.fit_transform(x["gender"])
x["hospital_admit_source"] = pre.fit_transform(x["hospital_admit_source"])
x["icu_admit_source"] = pre.fit_transform(x["icu_admit_source"])
x["icu_stay_type"] = pre.fit_transform(x["icu_stay_type"])
x["icu_type"] = pre.fit_transform(x["icu_type"])
x["apache_3j_bodysystem"] = pre.fit_transform(x["apache_3j_bodysystem"])
x["apache_2_bodysystem"] = pre.fit_transform(x["apache_2_bodysystem"])
x = x.astype(int)

y = df["hospital_death"]


test = pd.read_csv("unlabeled.csv")
encounter_id = test["encounter_id"]


test = test.drop(["albumin_apache","bilirubin_apache","fio2_apache","paco2_apache","pao2_apache","urineoutput_apache","d1_diasbp_invasive_max","d1_diasbp_invasive_min","d1_mbp_invasive_max","d1_mbp_invasive_min","d1_sysbp_invasive_max","d1_sysbp_invasive_min","h1_diasbp_invasive_max","h1_diasbp_invasive_min","h1_mbp_invasive_max","h1_mbp_invasive_min","d1_albumin_max","d1_albumin_min","d1_bilirubin_max","d1_bilirubin_min","d1_inr_max","d1_inr_min","d1_lactate_max","h1_albumin_max","h1_albumin_min","h1_bilirubin_max","h1_bilirubin_min"],axis=1)
test = test.drop(["h1_sysbp_invasive_max","h1_sysbp_invasive_min","d1_lactate_min","h1_bun_max","h1_bun_min","h1_calcium_max","h1_calcium_min","h1_creatinine_max","h1_creatinine_min","h1_glucose_max","h1_glucose_min","h1_hco3_max","h1_hco3_min","h1_hemaglobin_max","h1_hemaglobin_min"],axis=1)
test = test.drop(["h1_hematocrit_max","h1_hematocrit_min","h1_lactate_max","h1_lactate_min","h1_platelets_max","h1_platelets_min","h1_potassium_max","h1_potassium_min","h1_sodium_max","h1_sodium_min","h1_wbc_max","h1_wbc_min","d1_arterial_pco2_max","d1_arterial_pco2_min","d1_arterial_ph_min","d1_arterial_po2_max","d1_arterial_po2_min"],axis=1)
test = test.drop(["encounter_id","patient_id","hospital_id","hospital_death","d1_pao2fio2ratio_max","d1_pao2fio2ratio_min","h1_arterial_pco2_max","h1_arterial_pco2_min","h1_arterial_ph_max","h1_arterial_ph_min","h1_arterial_po2_max","h1_arterial_po2_min","h1_pao2fio2ratio_max","h1_pao2fio2ratio_min"],axis=1)
test = test.drop(["ph_apache","ventilated_apache","d1_diasbp_noninvasive_max","d1_calcium_max"],axis=1)
fill = SimpleImputer(missing_values=np.NAN,strategy="most_frequent")
fill  = fill.fit_transform(test)
test = pd.DataFrame(fill,columns=test.columns)

pre = pr.LabelEncoder()
test["ethnicity"] = pre.fit_transform(test["ethnicity"])
test["gender"] = pre.fit_transform(test["gender"])
test["hospital_admit_source"] = pre.fit_transform(test["hospital_admit_source"])
test["icu_admit_source"] = pre.fit_transform(test["icu_admit_source"])
test["icu_stay_type"] = pre.fit_transform(test["icu_stay_type"])
test["icu_type"] = pre.fit_transform(test["icu_type"])
test["apache_3j_bodysystem"] = pre.fit_transform(test["apache_3j_bodysystem"])
test["apache_2_bodysystem"] = pre.fit_transform(test["apache_2_bodysystem"])
test = test.astype(int)
#print(x.shape)
#print(test.shape)

#original score
#test  0.8406169665809768
#train  0.8935006435006435

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.20,random_state=0)
#Using RFE
#model = xgb.XGBClassifier()
#rfe = RFE(model,n_features_to_select=109)
#rfe = rfe.fit(x_train,y_train)
#
#new  = pd.DataFrame({"columns":x.columns,"ranking":rfe.ranking_,"selected":rfe.support_})
#
#new = new.astype(str)
#new = new[new["selected"].str.contains("False")]
#col = new["columns"].values
#print(col)

"""
#Optimizing Parameters
model = xgb.XGBClassifier()
param = {"max_depth":[3,4,5,8,9],"subsample":[0.6,0.7,1.0],"colsample_bytree":[0.6,0.9,1.0],"min_child_weight":[1,2,3,6,1],"learning_rate":[0.01,0.1,1.0,0.3,0.5]}
search = BayesSearchCV(model,param,scoring="accuracy")
search = search.fit(x_train,y_train)
print(search.best_params_)
"""
#Training my model


model = xgb.XGBClassifier(colsample_bytree=0.1,learning_rate=0.1,max_depth=3,subsample=0.8,min_child_weight=15,n_estimators=559)
model = model.fit(x_train,y_train)
tmodel =  model.score(x_test,y_test)
trmodel = model.score(x_train,y_train)
print("test",tmodel)
print("train",trmodel)



#ypred = model.predict(test)
#submission = {"encounter_id":encounter_id,"hospital_death":ypred}
#submission = pd.DataFrame(submission)
#kaggle_submit = submission.to_csv("kaggle_submit.csv",index = False)
#print(kaggle_submit)








