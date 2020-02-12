import pandas as pd
from  matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


df = pd.read_csv("data.csv")

df = df.drop(["albumin_apache","bilirubin_apache","fio2_apache","paco2_apache","pao2_apache","urineoutput_apache","d1_diasbp_invasive_max","d1_diasbp_invasive_min","d1_mbp_invasive_max","d1_mbp_invasive_min","d1_sysbp_invasive_max","d1_sysbp_invasive_min","h1_diasbp_invasive_max","h1_diasbp_invasive_min","h1_mbp_invasive_max","h1_mbp_invasive_min","d1_albumin_max","d1_albumin_min","d1_bilirubin_max","d1_bilirubin_min","d1_inr_max","d1_inr_min","d1_lactate_max","h1_albumin_max","h1_albumin_min","h1_bilirubin_max","h1_bilirubin_min"],axis=1)
df = df.drop(["h1_sysbp_invasive_max","h1_sysbp_invasive_min","d1_lactate_min","h1_bun_max","h1_bun_min","h1_calcium_max","h1_calcium_min","h1_creatinine_max","h1_creatinine_min","h1_glucose_max","h1_glucose_min","h1_hco3_max","h1_hco3_min","h1_hemaglobin_max","h1_hemaglobin_min"],axis=1)
df = df.drop(["h1_hematocrit_max","h1_hematocrit_min","h1_lactate_max","h1_lactate_min","h1_platelets_max","h1_platelets_min","h1_potassium_max","h1_potassium_min","h1_sodium_max","h1_sodium_min","h1_wbc_max","h1_wbc_min","d1_arterial_pco2_max","d1_arterial_pco2_min","d1_arterial_ph_min","d1_arterial_po2_max","d1_arterial_po2_min"],axis=1)
df = df.drop(["d1_pao2fio2ratio_max","d1_pao2fio2ratio_min","h1_arterial_pco2_max","h1_arterial_pco2_min","h1_arterial_ph_max","h1_arterial_ph_min","h1_arterial_po2_max","h1_arterial_po2_min","h1_pao2fio2ratio_max","h1_pao2fio2ratio_min"],axis=1)

x = df.drop(["hospital_death"],axis=1)
y = df["hospital_death"]

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.20,random_state=0)

model = xgb.XGBClassifier()
model = model.fit(x_train,y_train)
model = model.score(x_test,y_test)
print(model)


