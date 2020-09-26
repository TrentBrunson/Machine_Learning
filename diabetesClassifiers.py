#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

#%%
diabetesDF: pd.DataFrame = pd.read_csv('./data/diabetes.csv') 

attributeDF: pd.DataFrame = diabetesDF.drop("Outcome", axis=1)

labels: pd.Series = diabetesDF["Outcome"]

#%%
attributeTraining, attributeTesting, labelTraining, labelTesting = train_test_split(attributeDF,labels)
# %%

logisticRegClassifier: LogisticRegression = LogisticRegression(solver='lbfgs', max_iter=200)
# %%
regModel: LogisticRegression =  logisticRegClassifier.fit(attributeTraining,labelTraining)
# %%
print(regModel.score(attributeTesting,labelTesting))
# %%

predictedLabels: np.ndarray = regModel.predict(attributeTesting)

# %%
confusionMatrixDF: pd.DataFrame = pd.DataFrame(confusion_matrix(labelTesting,predictedLabels),index=['Actual +','Actual -'],columns=['Predicted +', 'Predicted -'])
# %%
print(confusionMatrixDF)
# %%
print(classification_report(labelTesting,predictedLabels))
# %%
