#%%
from sklearn.datasets import fetch_lfw_people
from PIL import Image
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# %%
faceDataSet = fetch_lfw_people(min_faces_per_person=50)

# %%
print(faceDataSet.DESCR)
#%%
faceDataSet
# %%
fig, axis = plt.subplots(6,6)
for i,iAxis in enumerate(axis.flat):
    iAxis.imshow(faceDataSet.images[i])
    iAxis.set(xlabel=faceDataSet.target_names[i])
# %%
len(faceDataSet.images)
#%%
for name in faceDataSet.target_names:
    print(name)
#%%
XTrain, XTest, yTrain, yTest = train_test_split(faceDataSet.data, faceDataSet.target)
# %%
pca: PCA = PCA (n_components=120)

svm: SVC = SVC ()

reduceDimensionPipeline = make_pipeline(pca,svm)
#%%
paramsToTestDict: dict = {
    "svc__C" : [1.0,5.0,10.0,50.0,100.0],
    "svc__gamma" : [0.0001,0.0005,0.1,0.5,1.0]
    }

# %%
paramSearchGrid: GridSearchCV = GridSearchCV(estimator = reduceDimensionPipeline, param_grid=paramsToTestDict)
# %%
paramSearchGrid.fit(XTrain,yTrain)
# %%
