#%%
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

#%%
populationSize = 4900

sensitivity = 0.72 #tp / (tp + fn)
specificity = 0.93 #tp / (tp + fp)

probDisease = 0.1
probDontHaveDisease = 1 - probDisease

riskOfDeath = 0.0005
riskOfSexDisfunction = 0.01
riskOfOtherComplications = 0.21


truePositive = populationSize * probDisease * specificity
trueNegative = populationSize * probDontHaveDisease * specificity
falsePositive = populationSize * probDisease * ( 1 - sensitivity)
falseNegative = populationSize * probDontHaveDisease * ( 1 - 0.93)
#%%
class Patient:
    def __init__(self,patientId: int, testPositive: bool, operation: bool, death: bool, sexDysfunction: bool, otherComplications: bool):
        self.patientId = patientId
        self.testPositive = testPositive
        self.operation = operation
        self.death = death
        self.sexDysfunction = sexDysfunction
        self.otherComplications = otherComplications
    
    def __repr__(self):
        return f"Patient:{self.patientId}"

#%%
def randomAttributeBool(num: float) -> bool:
    scalingFactor: int = 100000000000
    randNum = random.randint(1,scalingFactor)
    scaledNum = num*scalingFactor

    if scaledNum >= randNum:
        return True
    else:
        return False

#%%
patientList = []
monteCarloGenerateAttributes = []
monteCarloGenerateLabels = []

for i in range(0, populationSize):
    postiveTest = randomAttributeBool(probDisease)

    patient = Patient(
        patientId=i,
        testPositive=postiveTest,
        operation=postiveTest,
        death=randomAttributeBool(riskOfDeath),
        sexDysfunction=randomAttributeBool(riskOfSexDisfunction),
        otherComplications=randomAttributeBool(riskOfOtherComplications)
    )

    patientList.append(patient)

    monteCarloGenerateAttributes.append([int(patient.patientId),int(patient.operation),int(patient.death),int(patient.sexDysfunction), int(patient.otherComplications)])

    if patient.testPositive:
        if patient.death:
            monteCarloGenerateLabels.append(1)
        else:
            if patient.sexDysfunction and patient.otherComplications:
                monteCarloGenerateLabels.append(2)
            elif patient.sexDysfunction:
                monteCarloGenerateLabels.append(3)
            elif patient.otherComplications:
                monteCarloGenerateLabels.append(4)
            else:
                monteCarloGenerateLabels.append(5)
    else:
        monteCarloGenerateLabels.append(0)
# %%
print(patientList)
# %%
trainAttr,testAttr,trainLabel,testLabel = train_test_split(monteCarloGenerateAttributes,monteCarloGenerateLabels)
# %%
decisionTree: DecisionTreeClassifier = DecisionTreeClassifier()
decisionTree.fit(trainAttr,trainLabel)


# %%
print(decisionTree.score(testAttr,testLabel))
# %%
predictLabels = decisionTree.predict(testAttr)
#%%
print(classification_report(testLabel,predictLabels))
# %%
