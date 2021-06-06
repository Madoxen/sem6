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
    x, y, test_size=0.40, random_state=1)

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

gender.automf(2, names=["Male", "Female"])
age.automf(4, names=["0-18", "18-35", "35-65", "65+"])
hypertension.automf(2, names=["No", "Yes"])
heart_disease.automf(2, names=["No", "Yes"])
ever_married.automf(2, names=["No", "Yes"])
work_type.automf(5, names=['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'])
Residence_type.automf(2, names=["Urban", "Rural"])
avg_glucose_level.automf(3, names=["Low", "Medium", "High"])
bmi.automf(4, names=['underweight', 'normal', 'overweight', 'obese', 'extremely_obese'])
smoking_status.automf(4, names=['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
stroke.automf(2, names=["No", "Yes"])

rule1 = ctrl.Rule(heart_disease['Yes'] &
                  avg_glucose_level['High'], stroke['Yes'])
rule2 = ctrl.Rule(heart_disease['Yes'] &
                  avg_glucose_level['Medium'], stroke['Yes'])
rule3 = ctrl.Rule(heart_disease['Yes'], stroke['Yes'])
rule4 = ctrl.Rule(smoking_status['smokes'], stroke['Yes'])
rule5 = ctrl.Rule(smoking_status['never smoked'], stroke['No'])
rule6 = ctrl.Rule(heart_disease['No'], stroke['No'])
rule7 = ctrl.Rule(heart_disease['No'] &
                  smoking_status['never smoked'], stroke['No'])
rule8 = ctrl.Rule(heart_disease['Yes'] &
                  smoking_status['smokes'], stroke['Yes'])
rule9 = ctrl.Rule(heart_disease['Yes'] &
                  smoking_status['formerly smoked'], stroke['Yes'])
rule10 = ctrl.Rule(heart_disease['Yes'], stroke['Yes'])
rule11 = ctrl.Rule(bmi['extremely_obese'], stroke['Yes'])
rule12 = ctrl.Rule(bmi['underweight'], stroke['No'])
rule13 = ctrl.Rule(heart_disease['No'], stroke['No'])
rule14 = ctrl.Rule(bmi['underweight'] & age['0-18'], stroke['No'])
rule15 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['High'], stroke['Yes'])
rule16 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['Medium'], stroke['Yes'])
rule17 = ctrl.Rule(heart_disease['Yes'], stroke['Yes'])
rule18 = ctrl.Rule(work_type['Never_worked'], stroke['Yes'])
rule19 = ctrl.Rule(work_type['Private'], stroke['No'])
rule20 = ctrl.Rule(Residence_type['Urban'], stroke['No'])
rule21 = ctrl.Rule(age['0-18'] & work_type['Private'], stroke['No'])
rule22 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['High'], stroke['Yes'])
rule23 = ctrl.Rule(age['65+'] & avg_glucose_level['High'], stroke['Yes'])
rule24 = ctrl.Rule(heart_disease['Yes'], stroke['Yes'])
rule25 = ctrl.Rule(age['65+'], stroke['Yes'])
rule26 = ctrl.Rule(age['0-18'], stroke['No'])
rule27 = ctrl.Rule(hypertension['No'], stroke['No'])
rule28 = ctrl.Rule(gender['Male'] & ever_married['No'], stroke['No'])
rule29 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['High'], stroke['Yes'])
rule30 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['Medium'], stroke['Yes'])
rule30 = ctrl.Rule(heart_disease['Yes'] &
                   avg_glucose_level['Medium'], stroke['Yes'])
rule31 = ctrl.Rule(Residence_type['Rural'] | bmi['normal'], stroke['No'])
rule32 = ctrl.Rule(gender['Female'] | age['0-18'] | hypertension['No'] | heart_disease['No'] |
                   ever_married['No'] | work_type['Private'] | Residence_type['Urban'] | bmi['underweight'], stroke['No'])
rule33 = ctrl.Rule(gender['Male'] | age['65+'] | hypertension['Yes'] | heart_disease['Yes'] |
                   ever_married['Yes'] | work_type['Never_worked'] | Residence_type['Rural'] | bmi['extremely_obese'], stroke['Yes'])

stroke_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
                                  rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33])
stroke = ctrl.ControlSystemSimulation(stroke_ctrl)


stroke.input['heart_disease'] = 0.0
stroke.input['avg_glucose_level'] = 0.0
stroke.input['bmi'] = 0.0
stroke.input['age'] = 0.0
stroke.input['smoking_status'] = 0.0
stroke.input['work_type'] = 0.0
stroke.input['gender'] = 0.0
stroke.input['Residence_type'] = 0.0
stroke.input['hypertension'] = 0.0
stroke.input['ever_married'] = 0.0


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
    print(stroke.output['stroke'])
    if stroke.output['stroke'] > treshold:
        val = 1

    if val == 1:
        if val == true_val:
            TP += 1  # predicted
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
tpr = TP / (TP + FN)
tnr = TN / (TN + FP)
acc = TP+TN / (y_test.count())
ppv = TP / (TP + FP)

print("Sensitivity (TPR) " + str(tpr))
print("Specificity (TNR) " + str(tnr))
print("Accuracy " + str(acc))
print("Precision (PPV) " + str(ppv))
print("F1 " + str(2.0*ppv*tpr/(ppv+tpr)))


# testing phase
# go through the test set
