#%%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
#%%
digits = datasets.load_digits()

#%%
print(digits.DESCR)

#%%
len(digits.images)

#%%
numberOfSamples = len(digits.images)
# %%
x = digits.images.reshape((numberOfSamples, -1))
# %%
x.shape
# %%
digits.images.shape
# %%
y = digits.target
# %%
len(y)
# %%
trainAttr,testattr, trainLabel, testLabel = train_test_split(x,y)
# %%
bestOverallScore = 0
bestParameter = {"bestNEstimator":None, "bestMaxDepth":None} 
#%%
n_estimators = [5,10,20,50,100,1000]
max_depth = [5,10,20,50,100,1000,None]

for n in n_estimators:
    for depth in max_depth:
        randomForest: RandomForestClassifier = RandomForestClassifier(n_estimators=n, max_depth=depth)
        randomForest.fit(trainAttr, trainLabel)
        score = randomForest.score(testattr, testLabel)
        if score > bestOverallScore:
            bestOverallScore = score
            bestParameter['bestNEstimator'] = n
            bestParameter['bestMaxDepth'] = depth

# %%
print(
    f"\nBest Score: {bestOverallScore}\n\n"
    f"Best Estimator: {bestParameter['bestNEstimator']}\n"
    f"Best Depth: {bestParameter['bestMaxDepth']}\n"
)
# %%
print(bestParameter)
# %%
