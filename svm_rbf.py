#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from hypopt import GridSearch
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


#filepaths
base = 'UCI HAR Dataset/'
f_x_train = base + 'train/X_train.txt'
f_x_test = base + 'test/X_test.txt'
f_y_train = base + 'train/Y_train.txt'
f_y_test = base + 'test/Y_test.txt'


# In[3]:


def readX(filename):
    # Using readlines() 
    file = open(filename, 'r') 
    Lines = file.readlines() 
    vec_arr = []
    for line in Lines: 
        vec = []
        line = line.strip()
        for word in line.split():
            vec.append(float(word))
        vec_arr.append(vec)
    X = np.array(vec_arr)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# In[4]:


def readY(filename):
    # Using readlines() 
    file = open(filename, 'r') 
    Lines = file.readlines() 
    vec_arr = []
    for line in Lines:
        line = line.strip()
        vec_arr.append(int(line))
    return np.array(vec_arr)


# In[5]:


X_train = readX(f_x_train)
X_test = readX(f_x_test)
Y_train = readY(f_y_train)
Y_test = readY(f_y_test)
data = np.append(X_train, Y_train.reshape([Y_train.shape[0], 1]), axis = 1)


# In[6]:


classes, counts = np.unique(Y_train, return_counts=True)
plt.bar(classes, counts)
plt.show()


# In[7]:


pca = PCA(n_components=20)
pca.fit(X_train)
print(sum(pca.explained_variance_ratio_))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[8]:


data_pca = np.append(X_train_pca, Y_train.reshape([Y_train.shape[0], 1]), axis = 1)


# In[9]:


def get_acc(X, Y, model):
    Y_pred = model.predict(X)
    corr = 0.0
    for i in range(Y_pred.shape[0]):
        if Y_pred[i] == Y[i]:
            corr+=1
    return (corr/Y.shape[0])


# In[11]:


clf = SVC()
parameters=[{'gamma': [0.01, 0.001, 0.0001, 0.00001], 'C': [10, 100, 1000,10000]}]


# In[15]:


model=GridSearchCV(clf,parameters,n_jobs=-1,cv=10,verbose=10)
model.fit(X_train.tolist(),Y_train.tolist())


# In[16]:


get_acc(X_test, Y_test, model)


# In[17]:


#model.cv_results_
model.best_params_


# In[51]:


def plot_svm_grid(model):    
    gammas = [0.01, 0.001, 0.0001, 0.00001]
    Cs = [10, 100, 1000,10000]
    scores = model.cv_results_['mean_test_score'].reshape(len(gammas),len(Cs))
    print(scores)
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.inferno)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gammas)), gammas)
    plt.yticks(np.arange(len(Cs)), Cs)
    plt.title('Grid Search Accuracy')
    plt.show()


# In[12]:


def cross_val_grid(data = data, model = 'svm'):
    best_model = None
    best_acc = 0
    model_arr = [] #10 models from grid search
    for i in range(10): #10 times
        np.random.shuffle(data) #shuffle every time
        clf = SVC()
        parameters=[{'gamma': [0.01, 0.001, 0.0001, 0.00001], 'C': [10, 100, 1000,10000]}]
        model=GridSearchCV(clf,parameters,n_jobs=-1,cv=10,verbose=10)
        model.fit(data[:,:-1].tolist(),data[:,-1].tolist())
        acc = model.best_score_
        model_arr.append(model)
        print(i, "run accuracy =", acc, ", params = ", model.best_params_)
        if(acc > best_acc):
            best_acc = acc
            best_model = model
    print("Best accuracy =", best_acc, ", params = ", best_model.best_params_)
    return best_model


# In[27]:


best_svm = cross_val_grid()


# In[32]:


get_acc(X_test, Y_test, best_svm)


# In[52]:


plot_svm_grid(best_svm)


# In[13]:


best_svm_pca = cross_val_grid(data = data_pca)


# In[15]:


plot_svm_grid(best_svm_pca)


# In[16]:


get_acc(X_test_pca, Y_test, best_svm_pca)


# In[46]:


# DT on raw data
dt = DecisionTreeClassifier()
dt = dt.fit(X_train,Y_train)


# In[47]:


get_acc(X_test, Y_test, dt)


# In[45]:


# DT on pca
dt_pca = DecisionTreeClassifier()
dt_pca = dt_pca.fit(X_train_pca,Y_train)


# In[48]:


get_acc(X_test_pca, Y_test, dt_pca)


# In[38]:


best_svm = SVC(C=100, gamma = 0.001)
best_svm.fit(X_train, Y_train)
acc = get_acc(X_test, Y_test, best_model)
print(acc)


# In[15]:


def print_conf(X, Y, model):
    activities = {1:'WALKING', 2:'WALKING_UPSTAIRS', 3:'WALKING_DOWNSTAIRS', 4:'SITTING', 5:'STANDING', 6:'LAYING'}
    Y_pred = model.predict(X)
    array = np.zeros((np.unique(Y).shape[0], np.unique(Y).shape[0])) # confusion matrix
    for i, y in enumerate(Y_pred):
        array[int(y)-1][int(Y[i])-1]+=1
    print(array)
    plt.figure(figsize=(10,7))
    
    sn.heatmap(array, annot=True, fmt='g', xticklabels = list(activities.values()), yticklabels = list(activities.values())) # font size
    plt.show()


# In[39]:


print_conf(X_test, Y_test,best_svm)


# In[41]:


print_conf(X_test_pca, Y_test,best_svm_pca)


# In[49]:


print_conf(X_test, Y_test, dt)


# In[50]:


print_conf(X_test_pca, Y_test,dt_pca)


# In[61]:


neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, Y_train)


# In[62]:


get_acc(X_test, Y_test, neigh)


# In[65]:


clf = KNeighborsClassifier()
parameters=[{'n_neighbors': [4,5,6]}]
neigh=GridSearchCV(clf,parameters,n_jobs=-1,cv=10,verbose=10)
neigh.fit(X_train,Y_train)


# In[70]:


print(neigh.best_params_)
get_acc(X_test, Y_test, neigh)


# In[72]:


print_conf(X_test, Y_test,neigh)


# In[67]:


clf_pca = KNeighborsClassifier()
parameters=[{'n_neighbors': [4,5,6]}]
neigh_pca=GridSearchCV(clf,parameters,n_jobs=-1,cv=10,verbose=10)
neigh_pca.fit(X_train_pca,Y_train)


# In[69]:


print(neigh_pca.best_params_)
get_acc(X_test_pca, Y_test, neigh_pca)


# In[71]:


print_conf(X_test_pca, Y_test,neigh_pca)


# In[74]:


clf_lin = SVC(kernel = 'linear')
parameters=[{'C': [10, 100, 1000,10000]}]
model_lin=GridSearchCV(clf_lin,parameters,n_jobs=8,cv=10,verbose=10)
model_lin.fit(X_train.tolist(),Y_train.tolist())


# In[75]:


print(model_lin.best_params_)
get_acc(X_test, Y_test, model_lin)


# In[77]:


print_conf(X_test, Y_test, model_lin)


# In[10]:


clf_lin_pca = SVC(kernel = 'linear')
parameters=[{'C': [10, 100, 1000, 10000]}]
model_lin_pca=GridSearchCV(clf_lin_pca,parameters,n_jobs=-1,cv=10,verbose=10)
model_lin_pca.fit(X_train_pca,Y_train)


# In[13]:


print(model_lin_pca.best_params_)
get_acc(X_test_pca, Y_test, model_lin_pca)


# In[16]:


print_conf(X_test_pca, Y_test, model_lin_pca)




