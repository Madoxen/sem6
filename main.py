from numpy.core.fromnumeric import mean
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

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
        'names': ['Low', 'Medium', 'High']
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
    'avg_glucose_level': {'Low': 0, 'Medium': 1, 'High': 2},
    'bmi': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3, 'extremely_obese': 4}
}

for (k, v) in value_mapper_dict.items():
    stroke[k] = stroke[k].map(v)


stroke['bmi'] = stroke['bmi'].fillna(
    mean(stroke['bmi'].dropna()))  # nans are assigned as mean


stroke = stroke.drop(
    columns=['gender', 'work_type', 'Residence_type', 'ever_married'])

x = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.95, random_state=1)

print(x_train)

age = ctrl.Antecedent(x_train["age"], 'age')
hypertension = ctrl.Antecedent(x_train["hypertension"], 'hypertension')
heart_disease = ctrl.Antecedent(x_train["heart_disease"], 'heart_disease')
avg_glucose_level = ctrl.Antecedent(
    x_train["avg_glucose_level"], 'avg_glucose_level')
bmi = ctrl.Antecedent(x_train["bmi"], 'bmi')
smoking_status = ctrl.Antecedent(x_train["smoking_status"], 'smoking_status')
stroke = ctrl.Consequent(y_train, 'stroke')


age.automf(4, names=["0-18", "18-35", "35-65", "65+"])
hypertension.automf(2, names=["No", "Yes"])
heart_disease.automf(2, names=["No", "Yes"])
avg_glucose_level.automf(3, names=["Low", "Medium", "High"])
bmi.automf(4, names=['underweight', 'normal',
                     'overweight', 'obese', 'extremely_obese'])
smoking_status.automf(
    4, names=['never smoked', 'Unknown', 'formerly smoked', 'smokes'])
stroke.automf(2, names=["No", "Yes"])


ruleset = [

    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['High'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['Medium'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'], stroke['Yes']),
    # ctrl.Rule(smoking_status['smokes'], stroke['Yes']),
    # ctrl.Rule(smoking_status['never smoked'], stroke['No']),
    # ctrl.Rule(heart_disease['No'], stroke['No']),
    # ctrl.Rule(heart_disease['No'] & smoking_status['never smoked'], stroke['No']),
    # ctrl.Rule(heart_disease['Yes'] & smoking_status['smokes'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'] & smoking_status['formerly smoked'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'], stroke['Yes']),
    # ctrl.Rule(bmi['extremely_obese'], stroke['Yes']),
    # ctrl.Rule(bmi['underweight'], stroke['No']),
    # ctrl.Rule(heart_disease['No'], stroke['No']),
    # ctrl.Rule(bmi['underweight'] & age['0-18'], stroke['No']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['High'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['Medium'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'], stroke['Yes']),
    # ctrl.Rule(age['0-18'], stroke['No']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['High'], stroke['Yes']),
    # ctrl.Rule(age['65+'] & avg_glucose_level['High'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'], stroke['Yes']),
    # ctrl.Rule(age['65+'], stroke['Yes']),
    # ctrl.Rule(age['0-18'], stroke['No']),
    # ctrl.Rule(hypertension['No'], stroke['No']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['High'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['Medium'], stroke['Yes']),
    # ctrl.Rule(heart_disease['Yes'] & avg_glucose_level['Medium'], stroke['Yes']),
    # ctrl.Rule(bmi['normal'], stroke['No']),
    # ctrl.Rule(age['0-18'] | hypertension['No'] | heart_disease['No'] | bmi['underweight'], stroke['No']),
    # ctrl.Rule(age['65+'] | hypertension['Yes'] | heart_disease['Yes'] | bmi['extremely_obese'], stroke['Yes']),
    # ctrl.Rule(age["35-65"] | hypertension["Yes"] | heart_disease["Yes"] | avg_glucose_level["Medium"] | bmi["obese"] | smoking_status["formerly smoked"], stroke['No'])

]

# prepare inverse value map
label_mapper_dict = {}
for k, v in value_mapper_dict.items():
    label_mapper_dict[k] = {}
    for key, value in v.items():
        label_mapper_dict[k][value] = key

print(label_mapper_dict)
label_mapper_dict['hypertension'] = {0: "No", 1: "Yes"}
label_mapper_dict['heart_disease'] = {0: "No", 1: "Yes"}
label_mapper_dict['stroke'] = {0: "No", 1: "Yes"}

# construct rules from training set
for index, row in x_train.iterrows():
    d = row.to_dict()
    val = y_train[index]
    ruleset.append(ctrl.Rule(
        age[label_mapper_dict['age'][d['age']]] &
        hypertension[label_mapper_dict['hypertension'][d['hypertension']]] &
        heart_disease[label_mapper_dict['heart_disease'][d['heart_disease']]] &
        avg_glucose_level[label_mapper_dict['avg_glucose_level'][d['avg_glucose_level']]] &
        bmi[label_mapper_dict['bmi'][d['bmi']]] &
        smoking_status[label_mapper_dict['smoking_status'][d['smoking_status']]], stroke[label_mapper_dict["stroke"][val]]
    ))



stroke_ctrl = ctrl.ControlSystem(ruleset)
stroke = ctrl.ControlSystemSimulation(stroke_ctrl)


stroke.input['heart_disease'] = 0.0
stroke.input['avg_glucose_level'] = 0.0
stroke.input['bmi'] = 0.0
stroke.input['age'] = 0.0
stroke.input['smoking_status'] = 0.0
stroke.input['hypertension'] = 0.0


TP = 0
TN = 0
FP = 0
FN = 0
# acceptance treshold, values bigger than this treshold are positive, rest are negative
treshold = 0.5

print(y_test.value_counts())

for index, row in x_test.iterrows():
    d = row.to_dict()
    val = 0
    true_val = y_test[index]
    print(d)
    print(true_val)
    stroke.inputs(d)

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


data = [[FN, TN], [TP, FP]]
ax = sns.heatmap(data, annot=True, fmt="d", linewidths=.5)
plt.show()
# testing phase
# go through the test set
