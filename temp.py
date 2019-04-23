# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
filepath = 'DScasestudy.txt'
rdf = pd.read_csv(filepath, sep='\t')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score
from sklearn import metrics

rdf.info()
rdf.describe()
rdf.head()
rdf.isnull().sum().any()
rdf.stack().unique()

X = rdf.drop('response', axis=1)
Y = rdf['response']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = .3, random_state = 2, stratify = Y)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# X = X.loc[:,(X.sum(axis=0) != len(X)) & (X.sum(axis=0) != 0)]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
#Scree plot
#https://plot.ly/ipython-notebooks/principal-component-analysis/ For each feature selection
#https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization


from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile,chi2
#X_vart = VarianceThreshold(threshold=0.1).fit_transform(X)
fs_VarT = VarianceThreshold(threshold=0.1)
fs_VarT.fit(X_train)
X_train_VarT = fs_VarT.transform(X_train)
X_test_VarT = fs_VarT.transform(X_test)
#Use descriptor to show remaining columns. Plot B4 and after PCA


X_kbest = SelectKBest(score_func=chi2, k=100)
X_kbest.fit(X)
X_selected = Xkbest.tranform(X)

X_Perc = SelectPercentile(percentile=50).fit(X,Y)
X_selected = X_perc.transform(X)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfeLoR = RFE(LogisticRegression(solver='saga',max_iter=1000),100)
#Sag model works well on large datasets but is sensitive to feature scaling. saga handles sparcity
rfeLoR.fit(X,Y)
rfeLoR.n_features_


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

m_RFERFC = RFECV(RandomForestClassifier(n_estimators=100), scoring='accuracy')
m_RFERFC.fit(X,Y) # returns model
X_RFERFC = m_RFERFC.predict(X)
m_RFERFC.score(X,Y)

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
m_lasso = SelectFromModel(LassoCV())
m_lasso.fit(X,Y)
m_lasso.transform(X).shape
X_lasso = m_lasso.transform(X)
m_lasso.get_params()
mask = m_lasso.get_support()
print(mask)
plt.matshow(mask.reshape(1,-1), cmap='gray_r')
X.columns[mask]
#Using CV helps reduce selection bias due to the observations in the training set


#X_test_selected = modelfit.transform(X_test)
#predmodel = logisticRegression()
#predmodel.fit(X_train,Y_train)
#print('The score on all features: {:.3f}'.format(predmodel.score(X_test,Y_test)))
#score = predmodel.fit(X_train_selected, y_train).score(X_test_selected,y_test)
#print('The score on all features: {:.3f}'.format(score))

from sklearn.ensemble import RandomForestClassifier
fs_SFM_RFC = SelectFromModel(RandomForestClassifier(n_estimators=100))
fs_SFM_RFC.fit(X_train,Y_train)
X_train_SFM_RFC = fs_SFM_RFC.transform(X_train)
X_train.shape
X_train_SFM_RFC.shape
#print('Dims of original data: {}, Dims of feature reduced data: {}'.format(X_train.shape,X_train_SFM_RFC.shape)
mask = fs_SFM_RFC.get_support()
print(mask)
X_train.columns[mask]






## build selector

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv = None,
                        n_jobs = 1, train_sizes=np.linespace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Score")
    
    plt.legend(loc="best")
    return plt

# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)
    


#Models
#Use Linear Regression as baseline for efficiency
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,31)
k_scores = []
for k in k_range:
    clf_knn = KNeighborsClassifier(n_neighbors=k) 
    scores = cross_val_score(clf_knn, X_train, Y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Mean CV Accuracy")
k_scores.index(max(k_scores)) #index 15 means k=16 is max
#Chosing higher values of K produces lower complexity models 

#Use GridSearchCV to find parameters easier
from sklearn.model_selection import GridSearchCV
#wt_opts = ['uniform', 'distance']
#param_grid = dict(n_neighbors=k_range,weights=wt_opts)
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy',n_jobs=-1)
grid.fit(X_train, Y_train)
plt.plot(k_range,grid.cv_results_['mean_test_score'])
grid.best_score_
grid.best_params_
grid.best_estimator_
grid.best_index_
#grid.cv_results_

grid.predict(X_train)
metrics.accuracy_score(Y_train,grid.predict(X_train))

y_pred = grid.predict(X_test)
metrics.accuracy_score(Y_test,y_pred)
Y_test.value_counts(normalize=True) #Null accuracy shows that predicting only most frequent class is close to the same probabilty
#Therefore model outcome is not great 
metrics.confusion_matrix(Y_test, y_pred)#Confusion matrix shows that we have classifying as 0 when the model should be predicted as 1!
#High false negative
#Tuning threshold to allow more positive responses to occur may be beneficial but without knowing reason for data not advisable 
y_pred_probs = grid.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_pred_probs)
plt.plot(fpr,tpr)
plt.title('roc curve')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
print('The AUC is: {:.4f}'.format(metrics.roc_auc_score(Y_test, y_pred_probs)))

def grid_clf_test(clf, param_grid, X_train,y_train,X_test,y_test,cv=10,scoring='accuracy'):
    grid = GridSearchCV(clf,param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train,y_train)
    print('The best params: ', grid.best_params_)
    grid.predict(X_train)
    print('Prediction accuracy to training class: ',metrics.accuracy_score(y_train,grid.predict(X_train)))
    y_pred = grid.predict(X_test)
    print('Prediction accuracy to test class: ',metrics.accuracy_score(y_test,y_pred))
    print('Value Counts for Null prediction: \n',y_test.value_counts(normalize=True))
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test,y_pred))
    y_pred_probs = grid.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_probs)
    plt.plot(fpr,tpr)
    plt.title('roc curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    print('The AUC is: {:.4f}'.format(metrics.roc_auc_score(y_test, y_pred_probs)))
    
#####Train Logistic Regression
#Faster way to train model is to use baeysian optimization over gridswearch
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
clf_LogR = LogisticRegression()
grid_clf_test(clf_LogR, param_grid,X_train,Y_train,X_test,Y_test)


####Random Forest
clf_RFC = RandomForestClassifier(n_estimators=1000)
param_grid={}
#Give RFC increase in estimator should only help. Let model auto tune max-features
grid_clf_test(clf_RFC,{},X_train,Y_train,X_test,Y_test)


####SVM
from sklearn.svm import SVC
param_grid={'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.001,0.01, 0.1, 1, 10, 100]}
clf_SVC = SVC(probability=True)
grid_clf_test(clf_SVC,param_grid,X_train,Y_train,X_test,Y_test)


####XGBoost
from xgboost import XGBClassifier
clf_XGB = XGBClassifier()
n_estimators=range(50,400,50)
max_depth=range(1,11,2)
param_grid=dict(n_estimators=n_estimators, max_depth=max_depth)
grid_clf_test(clf_XGB,param_grid,X_train,Y_train,X_test,Y_test)


#https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')
