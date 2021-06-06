from numpy.core.fromnumeric import mean
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import skfuzzy as fuzz
from skfuzzy import control as ctrl

stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

stroke = stroke.query("gender != 'Other'")  # oh no

stroke.drop('id', inplace=True, axis=1)
for c in stroke.columns:
    if c not in ['bmi', 'avg_glucose_level', 'age']:
        print(c, ":", Counter(stroke[c]))

nan = [np.nan]
print(stroke.query("bmi == @nan and smoking_status == 'Unknown'"))  # no corellation

nominal_dict = {
    'age': {
        'bins': [0, 18, 35, 65],
        'names': ['0-18', '18-35', '35-65', '65+']
    },
    'avg_glucose_level': {
        'bins': [0, 70, 130],
        'names': ['low', 'normal', 'high']
    },
    'bmi': {
        'bins': [0, 18.5, 25, 30, 35],
        'names': ['underweight', 'normal', 'overweight', 'obese', 'extremely_obese']
    }

}

for i in nominal_dict.keys():
    pair = nominal_dict[i]
    d = dict(enumerate(pair['names'], 1))
    bins = pair['bins']
    stroke[i] = np.vectorize(d.get)(np.digitize(stroke[i], bins))

value_mapper_dict = {
    'gender': {'Female': 0, 'Male': 1},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'Private': 3, 'Self-employed': 4, 'children': 0, 'Govt_job': 2, 'Never_worked': 1},
    'Residence_type': {'Urban': 0, 'Rural': 1},
    'smoking_status': {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3},
    'age': {'0-18': 0, '18-35': 1, '35-65': 2, '65+': 3},
    'avg_glucose_level': {'low': 0, 'normal': 1, 'high': 2},
    'bmi': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3, 'extremely_obese': 4}
}

for (k, v) in value_mapper_dict.items():
    stroke[k] = stroke[k].map(v)


stroke['bmi'] = stroke['bmi'].fillna(
    mean(stroke['bmi'].dropna()))  # nans are assigned as mean

x = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=1)

gender = ctrl.Antecedent(x_train["gender"], 'gender')
age = ctrl.Antecedent(x_train["age"], 'age')
hypertension = ctrl.Antecedent(x_train["hypertension"], 'hypertension')
heart_disease = ctrl.Antecedent(x_train["heart_disease"], 'heart_disease')
ever_married = ctrl.Antecedent(x_train["ever_married"], 'ever_married')
work_type = ctrl.Antecedent(x_train["work_type"], 'work_type')
Residence_type = ctrl.Antecedent(x_train["Residence_type"], 'Residence_type')
avg_glucose_level = ctrl.Antecedent(
    x_train["avg_glucose_level"], 'avg_glucose_level')
bmi = ctrl.Antecedent(x_train["bmi"], 'bmi')
smoking_status = ctrl.Antecedent(x_train["smoking_status"], 'smoking_status')
stroke = ctrl.Consequent(y_train, 'stroke')

gender.automf()
age.automf()
hypertension.automf()
heart_disease.automf()
ever_married.automf()
work_type.automf()
Residence_type.automf()
avg_glucose_level.automf()
bmi.automf()
smoking_status.automf()
stroke.automf()

rule1 = ctrl.Rule(heart_disease['good'] &
                  avg_glucose_level['good'], stroke['good'])
rule2 = ctrl.Rule(heart_disease['good'] &
                  avg_glucose_level['mediocre'], stroke['good'])
rule3 = ctrl.Rule(heart_disease['good'], stroke['good'])
rule4 = ctrl.Rule(smoking_status['good'], stroke['good'])
rule5 = ctrl.Rule(smoking_status['poor'], stroke['poor'])
rule6 = ctrl.Rule(heart_disease['poor'], stroke['poor'])
rule7 = ctrl.Rule(heart_disease['poor'] &
                  smoking_status['poor'], stroke['poor'])
rule8 = ctrl.Rule(heart_disease['good'] &
                  smoking_status['good'], stroke['good'])
rule9 = ctrl.Rule(heart_disease['good'] &
                  smoking_status['mediocre'], stroke['good'])
rule10 = ctrl.Rule(heart_disease['good'], stroke['good'])
rule11 = ctrl.Rule(bmi['good'], stroke['good'])
rule12 = ctrl.Rule(bmi['poor'], stroke['poor'])
rule13 = ctrl.Rule(heart_disease['poor'], stroke['poor'])
rule14 = ctrl.Rule(bmi['poor'] & age['poor'], stroke['poor'])
rule15 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['good'], stroke['good'])
rule16 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['mediocre'], stroke['good'])
rule17 = ctrl.Rule(heart_disease['good'], stroke['good'])
rule18 = ctrl.Rule(work_type['good'], stroke['good'])
rule19 = ctrl.Rule(work_type['poor'], stroke['poor'])
rule20 = ctrl.Rule(Residence_type['poor'], stroke['poor'])
rule21 = ctrl.Rule(age['poor'] & work_type['poor'], stroke['poor'])
rule22 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['good'], stroke['good'])
rule23 = ctrl.Rule(age['good'] & avg_glucose_level['good'], stroke['good'])
rule24 = ctrl.Rule(heart_disease['good'], stroke['good'])
rule25 = ctrl.Rule(age['good'], stroke['good'])
rule26 = ctrl.Rule(age['poor'], stroke['poor'])
rule27 = ctrl.Rule(hypertension['poor'], stroke['poor'])
rule28 = ctrl.Rule(gender['poor'] & ever_married['poor'], stroke['poor'])
rule29 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['good'], stroke['good'])
rule30 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['mediocre'], stroke['good'])
rule30 = ctrl.Rule(heart_disease['good'] &
                   avg_glucose_level['mediocre'], stroke['good'])
rule31 = ctrl.Rule(Residence_type['good'] | bmi['average'], stroke['poor'])
rule32 = ctrl.Rule(gender['poor'] | age['poor'] | hypertension['poor'] | heart_disease['poor'] |
                   ever_married['poor'] | work_type['poor'] | Residence_type['poor'] | bmi['poor'], stroke['poor'])
rule33 = ctrl.Rule(gender['good'] | age['good'] | hypertension['good'] | heart_disease['good'] |
                   ever_married['good'] | work_type['good'] | Residence_type['good'] | bmi['good'], stroke['good'])

stroke_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                                  rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33])
stroke = ctrl.ControlSystemSimulation(stroke_ctrl)


stroke.input['heart_disease'] = 1.0
stroke.input['avg_glucose_level'] = 1.0
stroke.input['bmi'] = 4.0
stroke.input['age'] = 2.0
stroke.input['smoking_status'] = 3.0
stroke.input['work_type'] = 4.0
stroke.input['gender'] = 1.0
stroke.input['Residence_type'] = 2.0
stroke.input['hypertension'] = 2.0
stroke.input['ever_married'] = 1.0


TP = 0
TN = 0
FP = 0
FN = 0
# acceptance treshold, values bigger than this treshold are positive, rest are negative
treshold = 0.5

print(y_test.value_counts())

for index, row in x_test.iterrows():
    d = row.to_dict()
    stroke.inputs(d)
    val = 0
    true_val = y_test[index]
    stroke.compute()
  ##  print(str(stroke.output['stroke'] > treshold) + " " + str(true_val))
    if stroke.output['stroke'] > treshold:
        val = 1

    if val == 1:
        if val == true_val:
            TP += 1 #predicted 
        else:
            FP += 1
    else:
        if val == true_val:
            TN += 1
        else:
            FN += 1


print("TP " + str(TP))
print("TN " + str(TN))
print("FP " + str(FP))
print("FN " + str(FN))

# testing phase
# go through the test set
