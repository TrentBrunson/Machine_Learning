#%%
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# %%
attrVector, labelVector = make_blobs(centers = 2, n_samples = 500, cluster_std=1.95)
# %%
plt.scatter(attrVector[:,0], attrVector[:,1], c=labelVector)
# %%
X_train, X_test, y_train, y_test = train_test_split(attrVector,labelVector)
# %%
svcInstance: SVC = SVC(kernel='rbf')
# %%
paramsToTestDict: dict = {
    "C" : [1.0,5.0,10.0,50.0,100.0],
    "gamma" : [0.0001,0.0005,0.1,0.5,1.0]
    }

paramSearchGrid: GridSearchCV = GridSearchCV(
    estimator = svcInstance, 
    param_grid=paramsToTestDict
    )


# svcInstance.fit(X_train, y_train)
# %%
paramSearchGrid.fit(X_train,y_train)
#%%
print(paramSearchGrid.best_params_)
#%%
predictedLabels = paramSearchGrid.predict(X_test)
#%%
print(classification_report(y_test,predictedLabels))

# %%
