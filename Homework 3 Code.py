# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 14:22:31 2023

@author: tomis
"""

#%% - import usuals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

#%% - set constants
test_size = 0.2
seed = 1234
np.random.seed(seed)

#%%
diabetes = pd.read_csv(r"D:\TechStuffs\Code\Machine Learning\Fundamentals of ML Class\Homeworks\Homework 3\diabetes.csv")

#%% - extract predictors and features

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)
shortened_columns = X.columns

# for graphing purposes
shortened_columns = [ 'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
       'Myocardial', 'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker',
       'HasHealthcare', 'CantAffordDoctor', 'GeneralHealth',
       'MentalHealth', 'PhysicalHealth', 'Stairs', 'Sex',
       'Age', 'Education', 'Income', 'Zodiac']

#%% - plot class imbalance
plt.figure()
plt.bar([0,1], y.value_counts(), color = ['b', 'r'], tick_label = ['No Diabetes', 'Diabetes'])
plt.ylabel('Count')
plt.title('Diabetes Dataset: Class Counts')
    
#%% 1 - Build a logistic regression model. Doing so: What is the best predictor of diabetes and
# what is the AUC of this model? 
columns = ['All'] + list(X.columns)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
model.fit(X_train, y_train)

index = ['Precision', 'Recall', 'AUROC', 'AUC', 'PR']
logisitic_regression_DF = pd.DataFrame(data=np.zeros((5, len(columns))), columns=columns, index=index)

"""
# the same - accuracy, pr, recall values changed slightly, but AUROC, AUC and PR are exactly the same
modifying threshold (using pred for AUC and AP):
Accuracy = 78.5%
Precision = 35.7%
Recall    = 64.7%
AUROC    = 0.818
AUC    = 0.818
PR    = 0.400
"""

# pred = model.predict(X_test)
pred_probs = model.predict_proba(X_test)

precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_probs[:,1])
f1_score = (2 * precision * recall) / (precision + recall)
idx = np.argmax(f1_score)
threshold = thresholds[idx]

pred = np.array([1 if prob > threshold else 0 for prob in pred_probs[:,1]])

accuracy = metrics.accuracy_score(y_test, pred) 
print("Accuracy = {:0.1f}%".format(accuracy * 100))

# The matrix of predictions and true values for each class.
conf_matrix = metrics.confusion_matrix(y_test, pred)

# Precision score.
precision = metrics.precision_score(y_test, pred)
logisitic_regression_DF.iloc[0,0] = precision
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(y_test, pred)
logisitic_regression_DF.iloc[1,0] = recall
print("Recall    = {:0.1f}%".format(100 * recall))

# ROC AUC Score
roc_auc = metrics.roc_auc_score(y_test, pred_probs[:,1])
logisitic_regression_DF.iloc[2,0] = roc_auc
print("AUROC    = {:0.3f}".format(roc_auc))

# AUC Score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs[:,1])
auc = metrics.auc(fpr, tpr)
logisitic_regression_DF.iloc[3,0] = auc
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs[:,1])
logisitic_regression_DF.iloc[4,0] = pr
print("PR    = {:0.3f}".format(pr))


#%%
y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)

def compute_logistic_drop(X, y, dropped_column, index):
    X = X.drop(dropped_column, axis=1)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
    model.fit(X_train, y_train)
    
    pred_probs = model.predict_proba(X_test)

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_probs[:,1])
    f1_score = (2 * precision * recall) / (precision + recall)
    idx = np.argmax(f1_score) #  0.45964088489137267
    threshold = thresholds[idx]
    
    pred = np.array([1 if prob > threshold else 0 for prob in pred_probs[:,1]])

    precision = metrics.precision_score(y_test, pred)
    logisitic_regression_DF.iloc[0,index] = precision

    # Recall score.
    recall = metrics.recall_score(y_test, pred)
    logisitic_regression_DF.iloc[1,index] = recall

    # ROC AUC Score
    roc_auc = metrics.roc_auc_score(y_test, pred_probs[:,1])
    logisitic_regression_DF.iloc[2,index] = roc_auc

    # AUC Score
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs[:,1])
    auc = metrics.auc(fpr, tpr)
    logisitic_regression_DF.iloc[3,index] = auc

    # Average Precision Score: summarizes PR-curve
    pr = metrics.average_precision_score(y_test, pred_probs[:,1])
    logisitic_regression_DF.iloc[4,index] = pr
    
    
for index, column in enumerate(X):
    compute_logistic_drop(X, y, column, index+1)


logisitic_regression_DF = logisitic_regression_DF.T

#%%
def plot_auc(df, y_bottom, y_top, model):
    df.index = ['All'] + shortened_columns
    auc = df[['AUROC']]

    auc = auc['AUROC'].sort_values()[:10]
    
    all_c = pd.Series({'All': [df.loc['All']['AUROC']]}).T
    all_c = all_c.rename('AUROC')
    auc = auc.append(all_c)
    
    plt.figure()
    plt.bar(list(auc.keys()), list(auc.values), label='AUC')
    plt.xticks(rotation='vertical')
    plt.ylim(y_bottom, y_top)
    plt.title("Drop in AUC for top 10 " + model +  " predictors")
    
def plot_pr(df, y_bottom, y_top, model):
    df.index = ['All'] + shortened_columns
    pr = df[['PR']]

    pr = pr['PR'].sort_values()[:10]
    
    all_c = pd.Series({'All': [df.loc['All']['PR']]}).T
    all_c = all_c.rename('PR')
    pr = pr.append(all_c)
    
    plt.figure()
    plt.bar(list(pr.keys()), list(pr.values), label='PR')
    plt.xticks(rotation='vertical')
    plt.ylim(y_bottom, y_top)
    plt.title("Drop in PR for top 10 " + model +  " predictors")
    
#%%
plot_auc(logisitic_regression_DF, 0.75, 0.83, "logistic regression")
plot_pr(logisitic_regression_DF, 0.2, 0.41, "logistic regression")

#%%
from sklearn import svm

#%% 2- Build a SVM. Doing so: What is the best predictor of diabetes and what is the AUC of
# this model? 

# linearsvc
X = np.array(X)
y = np.array(y)

index = ['Precision', 'Recall', 'AUROC', 'AUC', 'PR']
svm_DF = pd.DataFrame(data=np.zeros((5, len(columns))), columns=columns, index=index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = svm.LinearSVC(dual=False, class_weight='balanced')
model.fit(X_train, y_train)

"""
dual = False, LinearSVC, balanced class weight
Precision = 31.1%
Recall    = 76.3%
AUROC    = 0.818
AUC    = 0.818
PR    = 0.401
"""

pred = model.predict(X_test)
pred_probs = model.decision_function(X_test)

conf_matrix = metrics.confusion_matrix(y_test, pred)

# Precision score.
precision = metrics.precision_score(y_test, pred)
svm_DF.iloc[0,0] = precision
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(y_test, pred)
svm_DF.iloc[1,0] = recall
print("Recall    = {:0.1f}%".format(100 * recall))

# ROC AUC Score
roc_auc = metrics.roc_auc_score(y_test, pred_probs)
svm_DF.iloc[2,0] = roc_auc
print("AUROC    = {:0.3f}".format(roc_auc))

# AUC Score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
auc = metrics.auc(fpr, tpr)
svm_DF.iloc[3,0] = auc
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs)
svm_DF.iloc[4,0] = pr
print("PR    = {:0.3f}".format(pr))

#%% 
# try different kernels, not recommended for datset of this size

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)

def compute_svm_drop(X, y, dropped_column, index):
    X = X.drop(dropped_column, axis=1)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = svm.LinearSVC(dual=False, class_weight='balanced')
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    pred_probs = model.decision_function(X_test)

    precision = metrics.precision_score(y_test, pred)
    svm_DF.iloc[0,index] = precision

    # Recall score.
    recall = metrics.recall_score(y_test, pred)
    svm_DF.iloc[1,index] = recall

    # ROC AUC Score
    roc_auc = metrics.roc_auc_score(y_test, pred_probs)
    svm_DF.iloc[2,index] = roc_auc

    # AUC Score
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
    auc = metrics.auc(fpr, tpr)
    svm_DF.iloc[3,index] = auc

    # Average Precision Score: summarizes PR-curve
    pr = metrics.average_precision_score(y_test, pred_probs)
    svm_DF.iloc[4,index] = pr
    
for index, column in enumerate(X):
    compute_svm_drop(X, y, column, index+1)

svm_DF = svm_DF.T

#%%
plot_auc(svm_DF, 0.75, 0.83, "SVM")
plot_pr(svm_DF, 0.2, 0.41, "SVM")

#%%
from sklearn import tree

#%% 3 - Use a single, individual decision tree. Doing so: What is the best predictor of
# diabetes and what is the AUC of this model? 

X = np.array(X)
y = np.array(y)

index = ['Precision', 'Recall', 'AUROC', 'AUC', 'PR']
tree_DF = pd.DataFrame(data=np.zeros((5, len(columns))), columns=columns, index=index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = tree.DecisionTreeClassifier(class_weight='balanced', criterion='entropy')
model.fit(X_train, y_train)

"""
entropy -> higher AUC and PR scores
Precision = 30.3%
Recall    = 30.2%
AUROC    = 0.593
AUC    = 0.593
PR    = 0.191
"""

pred = model.predict(X_test)
pred_probs = model.predict_proba(X_test)[:, 1] # -> just 0 and 1, basically the same as pred

conf_matrix = metrics.confusion_matrix(y_test, pred)

# Precision score.
precision = metrics.precision_score(y_test, pred)
tree_DF.iloc[0,0] = precision
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(y_test, pred)
tree_DF.iloc[1,0] = recall
print("Recall    = {:0.1f}%".format(100 * recall))

# ROC AUC Score
roc_auc = metrics.roc_auc_score(y_test, pred_probs)
tree_DF.iloc[2,0] = roc_auc
print("AUROC    = {:0.3f}".format(roc_auc))

# AUC Score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
auc = metrics.auc(fpr, tpr)
tree_DF.iloc[3,0] = auc
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs)
tree_DF.iloc[4,0] = pr
print("PR    = {:0.3f}".format(pr))

#%%

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)

def compute_tree_drop(X, y, dropped_column, index):
    X = X.drop(dropped_column, axis=1)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = tree.DecisionTreeClassifier(class_weight='balanced', criterion='entropy')
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    precision = metrics.precision_score(y_test, pred)
    tree_DF.iloc[0,index] = precision

    # Recall score.
    recall = metrics.recall_score(y_test, pred)
    tree_DF.iloc[1,index] = recall

    # ROC AUC Score
    roc_auc = metrics.roc_auc_score(y_test, pred_probs)
    tree_DF.iloc[2,index] = roc_auc

    # AUC Score
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
    auc = metrics.auc(fpr, tpr)
    tree_DF.iloc[3,index] = auc

    # Average Precision Score: summarizes PR-curve
    pr = metrics.average_precision_score(y_test, pred_probs)
    tree_DF.iloc[4,index] = pr
    
for index, column in enumerate(X):
    compute_tree_drop(X, y, column, index+1)

tree_DF = tree_DF.T

#%%
plot_auc(tree_DF, 0.56, 0.60, "Tree")
plot_pr(tree_DF, 0.17, 0.20, "Tree")

#%%
from sklearn.ensemble import RandomForestClassifier

#%% 4 - Build a random forest model. Doing so: What is the best predictor of diabetes and
# what is the AUC of this model?

# hyperparemeter tuning

X = np.array(X)
y = np.array(y)

index = ['Precision', 'Recall', 'AUROC', 'AUC', 'PR']
forest_DF = pd.DataFrame(data=np.zeros((5, len(columns))), columns=columns, index=index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# max_samples = subset of data trained on, max_features = subset of features trained on
model = RandomForestClassifier(class_weight='balanced',n_estimators=100, max_samples=0.5, max_features=0.5, bootstrap=True, criterion='gini')
model.fit(X_train, y_train)

"""
cross validation -> gridsearchCV (n_estimators=100, max_samples=0.3, max_features=0.5, criterion=gini)
Precision = 53.1%
Recall    = 11.5%
AUROC    = 0.802
AUC    = 0.802
PR    = 0.386
"""

pred = model.predict(X_test)
pred_probs = model.predict_proba(X_test)[:, 1]
conf_matrix = metrics.confusion_matrix(y_test, pred)

# Precision score.
precision = metrics.precision_score(y_test, pred)
forest_DF.iloc[0,0] = precision
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(y_test, pred)
forest_DF.iloc[1,0] = recall
print("Recall    = {:0.1f}%".format(100 * recall))

# ROC AUC Score
roc_auc = metrics.roc_auc_score(y_test, pred_probs)
forest_DF.iloc[2,0] = roc_auc
print("AUROC    = {:0.3f}".format(roc_auc))

# AUC Score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
auc = metrics.auc(fpr, tpr)
forest_DF.iloc[3,0] = auc
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs)
forest_DF.iloc[4,0] = pr
print("PR    = {:0.3f}".format(pr))

#%%

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)

def compute_forest_drop(X, y, dropped_column, index):
    X = X.drop(dropped_column, axis=1)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = RandomForestClassifier(class_weight='balanced',n_estimators=100, max_samples=0.5, max_features=0.5, bootstrap=True, criterion='gini')
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    precision = metrics.precision_score(y_test, pred)
    forest_DF.iloc[0,index] = precision

    # Recall score.
    recall = metrics.recall_score(y_test, pred)
    forest_DF.iloc[1,index] = recall

    # ROC AUC Score
    roc_auc = metrics.roc_auc_score(y_test, pred_probs)
    forest_DF.iloc[2,index] = roc_auc

    # AUC Score
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
    auc = metrics.auc(fpr, tpr)
    forest_DF.iloc[3,index] = auc

    # Average Precision Score: summarizes PR-curve
    pr = metrics.average_precision_score(y_test, pred_probs)
    forest_DF.iloc[4,index] = pr
    
for index, column in enumerate(X):
    compute_forest_drop(X, y, column, index+1)

forest_DF = forest_DF.T

#%%
plot_auc(forest_DF, 0.76, 0.81, "Random Forest")
plot_pr(forest_DF, 0.32, 0.39, "Random Forest")

#%%
from sklearn.ensemble import AdaBoostClassifier

#%%  5 - Build a model using adaBoost. Doing so: What is the best predictor of diabetes and
# what is the AUC of this model?

X = np.array(X)
y = np.array(y)

index = ['Precision', 'Recall', 'AUROC', 'AUC', 'PR']
ada_DF = pd.DataFrame(data=np.zeros((5, len(columns))), columns=columns, index=index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# each tree just one level deep, but there are lots of them
# less estimators/less learning rate makes decision boundary less clearer
model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced'), algorithm="SAMME", n_estimators=300, learning_rate=1.5)
model.fit(X_train, y_train)

"""
GridSearchCV 
Precision = 29.7%
Recall    = 68.6%
AUROC    = 0.783
AUC    = 0.783
PR    = 0.341
"""

pred = model.predict(X_test)
pred_probs = model.predict_proba(X_test)[:, 1]
conf_matrix = metrics.confusion_matrix(y_test, pred)

# Precision score.
precision = metrics.precision_score(y_test, pred)
ada_DF.iloc[0,0] = precision
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(y_test, pred)
ada_DF.iloc[1,0] = recall
print("Recall    = {:0.1f}%".format(100 * recall))

# ROC AUC Score
roc_auc = metrics.roc_auc_score(y_test, pred_probs)
ada_DF.iloc[2,0] = roc_auc
print("AUROC    = {:0.3f}".format(roc_auc))

# AUC Score
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
auc = metrics.auc(fpr, tpr)
ada_DF.iloc[3,0] = auc
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs)
ada_DF.iloc[4,0] = pr
print("PR    = {:0.3f}".format(pr))

#%%

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)

def compute_ada_drop(X, y, dropped_column, index):
    X = X.drop(dropped_column, axis=1)
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced'), algorithm="SAMME", n_estimators=300, learning_rate=1.5)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]

    precision = metrics.precision_score(y_test, pred)
    ada_DF.iloc[0,index] = precision

    # Recall score.
    recall = metrics.recall_score(y_test, pred)
    ada_DF.iloc[1,index] = recall

    # ROC AUC Score
    roc_auc = metrics.roc_auc_score(y_test, pred_probs)
    ada_DF.iloc[2,index] = roc_auc

    # AUC Score
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs)
    auc = metrics.auc(fpr, tpr)
    ada_DF.iloc[3,index] = auc

    # Average Precision Score: summarizes PR-curve
    pr = metrics.average_precision_score(y_test, pred_probs)
    ada_DF.iloc[4,index] = pr
    
for index, column in enumerate(X):
    compute_ada_drop(X, y, column, index+1)

ada_DF = ada_DF.T
#%%
plot_auc(ada_DF, 0.73, 0.79, "AdaBoost")
plot_pr(ada_DF, 0.25, 0.35, "AdaBoost")
# interesting stuff --> dropping HighBP increases AUC/PR of model
# ChatGPT: Adaboost robust to noise, those 2 variables higly informative?
# dropping General Health performs worse than model with all -> so most beneficial
# dropping HighBP gives best performance -> so most detrimental
# why so different from other approaches

#%% Extra credit
y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# redo all models and plot AUC

# logistic regression
logistic_m = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
logistic_m.fit(X_train, y_train)
pred = logistic_m.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred), 4)
mcc = metrics.matthews_corrcoef(y_test, logistic_m.predict(X_test))
print("MCC", round(mcc, 4))
plt.plot(fpr,tpr, 'b',label="Logistic Regression, AUC="+str(auc))

# svm
svm_m = svm.LinearSVC(dual=False, class_weight='balanced')
svm_m.fit(X_train, y_train)
pred = svm_m.decision_function(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred), 4)
mcc = metrics.matthews_corrcoef(y_test, svm_m.predict(X_test))
print("MCC", round(mcc, 4))
plt.plot(fpr,tpr, 'r',label="SVM, AUC="+str(auc), alpha=0.5)

# tree
tree_m = tree.DecisionTreeClassifier(class_weight='balanced', criterion='entropy')
tree_m.fit(X_train, y_train)
pred = tree_m.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred), 4)
mcc = metrics.matthews_corrcoef(y_test, tree_m.predict(X_test))
print("MCC", round(mcc, 4))
plt.plot(fpr,tpr, 'g',label="Decision Tree, AUC="+str(auc))

# random forests
forest_m = RandomForestClassifier(class_weight='balanced',n_estimators=100, max_samples=0.5, max_features=0.5, bootstrap=True, criterion='gini')
forest_m.fit(X_train, y_train)
pred = forest_m.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred), 4)
mcc = metrics.matthews_corrcoef(y_test, forest_m.predict(X_test))
print("MCC", round(mcc, 4))
plt.plot(fpr,tpr, 'c',label="Random Forests, AUC="+str(auc))

# adaboost
ada_m = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1, class_weight='balanced'), algorithm="SAMME", n_estimators=300, learning_rate=1.5)
ada_m.fit(X_train, y_train)
pred = ada_m.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred), 4)
mcc = metrics.matthews_corrcoef(y_test, ada_m.predict(X_test))
print("MCC", round(mcc, 4))
plt.plot(fpr,tpr, 'k',label="AdaBoost, AUC="+str(auc))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curves for Each Model')
plt.legend()
#%%

# redo all models and plot PR
metrics.plot_precision_recall_curve(logistic_m, X_test, y_test, ax = plt.gca(),name = "Logistic Regression")
metrics.plot_precision_recall_curve(svm_m, X_test, y_test, ax = plt.gca(),name = "SVM")
metrics.plot_precision_recall_curve(tree_m, X_test, y_test, ax = plt.gca(),name = "Decision Tree")
metrics.plot_precision_recall_curve(forest_m, X_test, y_test, ax = plt.gca(),name = "Random Forest")
metrics.plot_precision_recall_curve(ada_m, X_test, y_test, ax = plt.gca(),name = "AdaBoost")
plt.legend(loc='upper right')
plt.title('Precision Recall Curves for Each Model')

#%%
# - entire correlation plot
import seaborn as sns
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(diabetes.corr(), annot = True, ax=ax)
plt.title("Correlation Matrix - Numbered")
plt.show()

#%% extra credit 2

# lifestyle choices and BMI
subset = ['Smoker', 'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker', 'BMI', 'Diabetes']
diabetes_sub = diabetes[subset]
bmi = diabetes['BMI']

#%%
diabetes.hist(bins=50,figsize = (20,20))

#%%
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(diabetes_sub.corr(), annot = True, ax=ax)
plt.title("Correlation Matrix - Numbered")
plt.show()

plt.hist(diabetes_sub['BMI'], bins=20)
plt.title('BMI Dist')
# most of the participants are overweight

#%%

diabetes_corr = diabetes.corr()['Diabetes']
#%%
# lifestyle choices

# 1 male, 2 female
sex = (diabetes['BiologicalSex'] == 1).map({True: 0, False:1}).rename('sex')

SES = diabetes[['EducationBracket', 'IncomeBracket']]
SES_Age = diabetes[['EducationBracket', 'IncomeBracket', 'AgeBracket']]
Age = diabetes['AgeBracket']
# further to-do: put BMI into these categories and then investigate for bad lifestyle choices and good lifestyle choices,
# controlling for BMI
X = pd.concat( (sex, Age), axis=1)
# X = sex
y = diabetes['Diabetes']

#%%

"""
just gender = array([[0.18437105]])
AUC    = 0.521
PR    = 0.148

gender + SES = array([[ 0.35150114, -0.17220943, -0.19547997]])
AUC    = 0.645
PR    = 0.217

gender + SES + Age = array([[ 0.35104909, -0.15382323, -0.17788006,  0.19351844]])
AUC    = 0.695
PR    = 0.243

gender + Age = array([[0.20040888, 0.20902482]])
AUC    = 0.647
PR    = 0.197
"""
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver = 'liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# pred = model.predict(X_test)
pred_probs = model.predict_proba(X_test)

precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_probs[:,1])
f1_score = (2 * precision * recall) / (precision + recall)
idx = np.argmax(f1_score)
threshold = thresholds[idx]

pred = np.array([1 if prob > threshold else 0 for prob in pred_probs[:,1]])

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probs[:,1])
auc = metrics.auc(fpr, tpr)
print("AUC    = {:0.3f}".format(auc))

# Average Precision Score: summarizes PR-curve
pr = metrics.average_precision_score(y_test, pred_probs[:,1])
print("PR    = {:0.3f}".format(pr))

#%%
sex_coefficients = [0.18437105, 0.35150114, 0.35104909,0.20040888]
labels = ['Sex', 'Sex + SES', 'Sex + SES + Age', 'Sex + Age']

bar_width = 0.8
# fig = plt.figure(figsize = (10, 10))

br1 = np.arange(len(sex_coefficients))

plt.bar(br1, sex_coefficients, width=bar_width)
ticks = [r for r in range(len(sex_coefficients))]
plt.xticks( ticks, labels )#, rotation=90)

plt.title("Sex Coefficients for different Controls")
plt.xlabel("Variables Controlled For")
plt.ylabel("Sex Coefficient")

plt.legend()
plt.show()
