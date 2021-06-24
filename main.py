from numpy.core.fromnumeric import mean
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
sns.set_theme()


def maxIndex(arr):
    maxVal = -float("inf")
    idx = 0
    for i in range(len(arr)):
        if arr[i] > maxVal:
            maxVal = arr[i]
            idx = i
    return idx


class trapezoidFuzzifier():
    def __init__(self, a, b, c, d, label):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.label = label
        super().__init__()

    def fuzzy(self, x):
        if x <= self.a:
            return 0
        if x > self.a and x <= self.b:
            return (x - self.a) / (self.b - self.a)
        if x > self.b and x < self.c:
            return 1
        if x >= self.c and x <= self.d:
            return (self.d - x) / (self.d - self.c)
        return 0


class triangleFuzzifier():

    def __init__(self, a, b, c, label):
        self.a = a
        self.b = b
        self.c = c
        self.label = label
        super().__init__()

    def fuzzy(self, x):
        if self.a < x and x <= self.b:
            return (x - self.a) / (self.b - self.a)
        if self.b < x and x < self.c:
            return (self.c - x) / (self.c - self.b)
        return 0


stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke = stroke.query("gender != 'Other'")  # oh no

stroke.drop('id', inplace=True, axis=1)
for c in stroke.columns:
    if c not in ['bmi', 'avg_glucose_level', 'age']:
        print(c, ":", Counter(stroke[c]))

nan = [np.nan]
# print(stroke.query("bmi == @nan and smoking_status == 'Unknown'"))  # no corellation


value_mapper_dict = {
    'gender': {'Female': 0, 'Male': 1},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'Private': 3, 'Self-employed': 4, 'children': 0, 'Govt_job': 2, 'Never_worked': 1},
    'Residence_type': {'Urban': 0, 'Rural': 1},
    'smoking_status': {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3},
}

for (k, v) in value_mapper_dict.items():
    stroke[k] = stroke[k].map(v)


stroke['bmi'] = stroke['bmi'].fillna(
    mean(stroke['bmi'].dropna()))  # nans are assigned as mean


x = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]


best = SelectKBest(score_func=chi2, k=10)
fit = best.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcols = pd.DataFrame(stroke.columns)
print(pd.concat([dfcols, dfscores], axis=1))


best = SelectKBest(score_func=chi2, k=10)
fit = best.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcols = pd.DataFrame(stroke.columns)
print(pd.concat([dfcols, dfscores], axis=1))

corrmat = stroke.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
# plot heat map
g = sns.heatmap(stroke[top_corr_features].corr(), annot=True, cmap="RdYlGn")


stroke = stroke.drop(
    columns=['gender', 'work_type', 'Residence_type', 'ever_married'])

x = stroke.iloc[:, :-1]
y = stroke.iloc[:, -1]

best = SelectKBest(score_func=chi2, k=6)
fit = best.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcols = pd.DataFrame(stroke.columns)
print(pd.concat([dfcols, dfscores], axis=1))

corrmat = stroke.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
# plot heat map
g = sns.heatmap(stroke[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


# create membership functions


ageMF = [
    trapezoidFuzzifier(0, 0, 10, 30, "lowAge"),
    triangleFuzzifier(20, 40, 50, "mediumAge"),
    trapezoidFuzzifier(40, 60, 80, 100, "highAge"),
]

glucoseMF = [
    trapezoidFuzzifier(0, 0, 10, 70, "lowGlucose"),
    triangleFuzzifier(60, 100, 120, "mediumGlucose"),
    trapezoidFuzzifier(110, 140, 400, 500, "highGlucose"),
]

bmiMF = [
    trapezoidFuzzifier(0, 0, 18.5, 25, "lowBmi"),
    triangleFuzzifier(24, 30, 35, "mediumBmi"),
    trapezoidFuzzifier(30, 35, 40, 45, "highBmi"),
]

smokingMF = [
    trapezoidFuzzifier(0, 0, 0.8, 1, "noSmoke"),  # never smoked, unknown
    trapezoidFuzzifier(2, 2, 2.8, 3, "smoke"),  # formerly smoked, smokes
]


# transform set to memberships
age = stroke["age"]
bmi = stroke["bmi"]
avg_glucose_level = stroke["avg_glucose_level"]
smoking_status = stroke["smoking_status"]

data = {}

for fuzzifier in ageMF:
    data[fuzzifier.label] = np.array(list(map(fuzzifier.fuzzy, age)))

for fuzzifier in bmiMF:
    data[fuzzifier.label] = np.array(list(map(fuzzifier.fuzzy, bmi)))

for fuzzifier in glucoseMF:
    data[fuzzifier.label] = np.array(
        list(map(fuzzifier.fuzzy, avg_glucose_level)))

for fuzzifier in smokingMF:
    data[fuzzifier.label] = np.array(
        list(map(fuzzifier.fuzzy, smoking_status)))

data = pd.DataFrame(data)
data = pd.concat([data, stroke["heart_disease"],
                  stroke["hypertension"]], axis=1)
data = data.dropna()


#x = df.iloc[:, :-1]
# x_train, x_test, y_train, y_test = train_test_split(
#   x, y, test_size=0.5, random_state=1)

# create ruleset from train_set
ruleLabels = [["lowAge", "mediumAge", "highAge"],
              ["lowBmi", "mediumBmi", "highBmi"],
              ["lowGlucose", "mediumGlucose", "highGlucose"],
              ["noSmoke", "smoke"],
              ["no_heart_disease", "heart_disease"],
              ["no_hypertension", "hypertension"]]


# create every possible rule permutation
rules = []

for ageRule in ruleLabels[0]:
    for bmiRule in ruleLabels[1]:
        for glucoseRule in ruleLabels[2]:
            for smokeRule in ruleLabels[3]:
                for heartRule in ruleLabels[4]:
                    for hypertensionRule in ruleLabels[5]:
                        rules.append(
                            [ageRule, bmiRule, glucoseRule, smokeRule, heartRule, hypertensionRule])

# for each rule permutation assign output based on points
pointMap = {
    "lowAge": 0,
    "mediumAge": 0.5,
    "highAge": 1,

    "lowBmi": 0,
    "mediumBmi": 0.5,
    "highBmi": 1,

    "lowGlucose": 0.25,
    "mediumGlucose": 0,
    "highGlucose": 1,

    "smoke": 1,
    "noSmoke": 0,

    "no_heart_disease": 0,
    "heart_disease": 1,
    "no_hypertension": 0,
    "hypertension": 1
}

point_treshold = 3.0  # points to die
rule_outputs = []
for rule in rules:
    points = 0
    for r in rule:
        points += pointMap[r]
    if points >= point_treshold:
        rule_outputs.append(True)  # dead
    else:
        rule_outputs.append(False)  # alive


#defuzzfy and classify
# for each input data row
predictions = []
for index, row in data.iterrows():
    # evaluate every single rule
    ruleProducts = []
    for rule in rules:
        # calculate product based on a rule
        prod = 1
        for r in rule:
            if r != "no_heart_disease" and r != "heart_disease" and r != "no_hypertension" and r != "hypertension":
                if row[r] != 0:
                    prod *= row[r]
            else:
                if r == "no_heart_disease" and row["heart_disease"] != 0:
                    prod *= 0.01  # penalty
                if r == "heart_disease" and row["heart_disease"] != 1:
                    prod *= 0.01  # penalty
                if r == "no_hypertension" and row["hypertension"] != 0:
                    prod *= 0.01  # penalty
                if r == "hypertension" and row["hypertension"] != 1:
                    prod *= 0.01  # penalty
        ruleProducts.append(prod)

    # after we got all rule products for one row, let's gather a prediction for given input row
    idx = maxIndex(ruleProducts)
    predictions.append(rule_outputs[idx])


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(predictions)):
    true_val = y.iloc[i]
    val = predictions[i]
    if true_val == True:
        if val == true_val:
            TP += 1
        else:
            FN += 1
    else:
        if val == true_val:
            TN += 1
        else:
            FP += 1


print("TP " + str(TP))
print("TN " + str(TN))
print("FP " + str(FP))
print("FN " + str(FN))
tpr = TP / (TP + FN)
tnr = TN / (TN + FP)
acc = TP+TN / (y.count())
ppv = TP / (TP + FP)

print("Sensitivity (TPR) " + str(tpr))
print("Specificity (TNR) " + str(tnr))
print("Accuracy " + str(acc))
print("Precision (PPV) " + str(ppv))
print("F1 " + str(2.0*ppv*tpr/(ppv+tpr)))


data = [[TP, FN], [FP, TN]]
ax = sns.heatmap(data, annot=True, fmt="d", linewidths=.5)
plt.show()
# testing phase
# go through the test set
