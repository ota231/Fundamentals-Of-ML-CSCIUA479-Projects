#%% - Import the usuals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

#%% - load data
tech_PD = pd.read_csv("techSalaries2017.csv")
state = 1234
test_size = 0.4

#%% - EDA
nulls = tech_PD.isnull().sum()

#%%
# NUMBER 1 -> preprocessing
# Using multiple linear regression: What is the best predictor of total annual
# compensation, how much variance is explained by this predictor vs. the full multiple
# regression model?

# drop 5) Base salary (in $), 6) Value of stock grants (in $) & 7) Bonus payments (in $)
# 4, 5 & 6 for 0 indexing
# also drop first 3 columns
tech_PD_regression = tech_PD.drop(tech_PD.iloc[:, 4:7], axis=1)
tech_PD_regression = tech_PD_regression.drop(tech_PD.iloc[:, :3], axis=1)

tech_PD_regression.dropna(inplace=True)

# drop race and education columns
tech_PD_regression = tech_PD_regression.drop(tech_PD_regression.iloc[:, 14:16], axis=1)


# drop one column from each dummy variable
# doctorate for education & multi-racial for race (just chose randomly)
tech_PD_regression = tech_PD_regression.drop(['Highschool', 'Race_Two_Or_More'], axis=1)

# drop "other" (106 rows) gender column
# reset indexes 
tech_PD_regression.reset_index(inplace=True)
other = list(np.where(tech_PD_regression['gender'] == 'Other')[0])
tech_PD_regression = tech_PD_regression.drop(labels=other, axis=0)
tech_PD_regression.reset_index(inplace=True)

# encode gender as categorical -> Male: 0, Female: 1
tech_PD_regression['Gender (Encoded)'] = pd.get_dummies(tech_PD_regression['gender'])['Female']
tech_PD_regression['Gender (Encoded)'] = tech_PD_regression['Gender (Encoded)'].astype('int64')
# drop original gender column
tech_PD_regression = tech_PD_regression.drop('gender', axis=1)

tech_PD_regression = tech_PD_regression.drop(tech_PD_regression.iloc[:, :2], axis=1)

# finally --> 21485, 17

#%% - relevant plots
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(tech_PD_regression.corr(), annot = True, ax=ax)
plt.title("Correlation Matrix - Numbered")
plt.show()

#%% linear regression function
from sklearn.linear_model import LinearRegression

# also get model coefficients and intercepts
def compute_linear_regression(X, y):
    y = np.array(y).reshape(-1, 1)
    X = np.array(X)
    model = LinearRegression().fit(X, y)
    error = mean_squared_error(y,model.predict(X),squared=False)
    R_2 = model.score(X, y)
    b_1 = model.coef_
    b_0 = model.intercept_
    return [R_2, b_1, b_0, error, model]

#%% NUMBER 1 - model
# Using multiple linear regression: What is the best predictor of total annual
# compensation, how much variance is explained by this predictor vs. the full multiple
# regression model?

y = tech_PD_regression['totalyearlycompensation']
X = tech_PD_regression.drop('totalyearlycompensation', axis=1)
features = X.columns

multiple_regression = compute_linear_regression( X , y )
multiple_regression_coefficients = multiple_regression[1]
# R_2 = 0.26509266407931054
R_2_per_column = []
for column in X:
    x = np.array(tech_PD_regression[column]).reshape(-1,1)
    print(column, compute_linear_regression(x, y)[0])
    R_2_per_column.append(compute_linear_regression(x, y)[0])
    # yearsofexperience 0.16086386858353963

experience_regression = compute_linear_regression( np.array(tech_PD_regression['yearsofexperience']).reshape(-1,1) , y )
    
# Education
# print(compute_linear_regression(tech_PD_regression.iloc[:, 3:7], y)[0]) # 0.03663050982626048
# Race
# print(compute_linear_regression(tech_PD_regression.iloc[:, 7:11], y)[0]) # 0.0027676922897398315

#%% NUMBER 1 - plot 
multiple_model = multiple_regression[-1]
multiple_R_2 = multiple_regression[0]
multiple_RMSE = multiple_regression[-2]

experience_model = experience_regression[-1]
experience_R_2 = experience_regression[0]
experience_RMSE = experience_regression[-2]

fig, axs = plt.subplots(2, sharey=True, figsize=(10,10))
plt.figure(figsize=(14, 12))

yHat_1 = multiple_model.predict(np.array(X))
axs[0].plot(yHat_1, np.array(y).reshape(-1, 1),'o', markersize=5) 
axs[0].set(xlabel="Prediction from Multiple model", ylabel="Actual Total Yearly Compensation")
axs[0].set_title("Mutiple Regression" + ", R^2 = {:.5f}".format(multiple_R_2) + " , RMSE = {:.0f}".format(multiple_RMSE))

yHat_2 = experience_model.predict( np.array(tech_PD_regression['yearsofexperience']).reshape(-1,1) )
axs[1].plot(yHat_2, np.array(y).reshape(-1, 1),'o', markersize=5) 
axs[1].set(xlabel="Prediction from Years of Experience", ylabel="Actual Total Yearly Compensation")
axs[1].set_title("Years of Experience" + ", R^2 = {:.5f}".format(experience_R_2) + " , RMSE = {:.0f}".format(experience_RMSE))



#%% - interlude
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
np.random.seed(state)

y = tech_PD_regression['totalyearlycompensation']
X = tech_PD_regression.drop('totalyearlycompensation', axis=1)
X = np.array(X)
y = np.array(y)



#%% NUMBER 2
# Using ridge regression to do the same as in 1): How does the model change or improve
# compared to OLS? What is the optimal lambda? 
# compare coefficents with OLS
# TO-DO: implement with RidgeCV

XTrain, XTest, yTrain, yTest = train_test_split(X, y.reshape(-1,1), test_size=test_size, random_state=state)
lambdas = np.linspace(0,500,5001)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]
for ii in range(len(lambdas)):
    ridgeModel = Ridge(alpha=lambdas[ii]).fit(XTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = mean_squared_error(yTest,ridgeModel.predict(XTest),squared=False)
    cont[ii,1] = error
    
ridge_optimal_lambda = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]
ridge_optimal_RMSE = np.min(cont[:,1])
optimal_ridgeModel = Ridge(alpha=ridge_optimal_lambda).fit(XTrain, yTrain)
print("Ridge R^2", optimal_ridgeModel.score(XTest, yTest))
ridge_coefficients = optimal_ridgeModel.coef_

plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Ridge regression: Optimal Lambda: {:.1f}'.format(ridge_optimal_lambda) + ', Minimum RMSE: {:.0f}'.format(ridge_optimal_RMSE) + \
          ", R^2 = {:.3f}".format(optimal_ridgeModel.score(XTest, yTest)))
plt.show()

#%% Number 2 - Graph
multiple_ridge = np.vstack((multiple_regression_coefficients, ridge_coefficients)).T

bar_width = 0.25
fig = plt.figure(figsize = (14, 10))

br1 = np.arange(len(multiple_ridge))
br2 = [x + bar_width for x in br1]

plt.bar(br1, multiple_ridge[:, 0], width=bar_width, label="multiple OLS regression")
plt.bar(br2, multiple_ridge[:, 1], width=bar_width, label="ridge regression")
ticks = [r + bar_width for r in range(len(multiple_ridge))]
plt.xticks( ticks, features, rotation='vertical')

plt.title("Coefficients for Multiple vs. Ridge")

plt.legend()
plt.show()

#%% - NUMBER 3
# Using Lasso regression to do the same as in 1): How does the model change now? How
# many of the predictor betas are shrunk to exactly 0? What is the optimal lambda now?
# implement with LassoCV

XTrain, XTest, yTrain, yTest = train_test_split(X, y.reshape(-1,1), test_size=test_size, random_state=state)
lambdas = np.linspace(0,1000,1001)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]
for ii in range(len(lambdas)):
    lassoModel = Lasso(alpha=lambdas[ii]).fit(XTrain, yTrain)
    cont[ii,0] = lambdas[ii]
    error = mean_squared_error(yTest,lassoModel.predict(XTest),squared=False)
    cont[ii,1] = error

lasso_optimal_lambda = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]
lasso_optimal_RMSE = np.min(cont[:,1])
optimal_lassoModel = Lasso(alpha=lasso_optimal_lambda).fit(XTrain, yTrain)
print("Lasso R^2", optimal_lassoModel.score(XTest, yTest))
lasso_coefficients = optimal_lassoModel.coef_
plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Lasso regression: Optimal Lambda: {:.1f}'.format(lasso_optimal_lambda) + ', Minimum RMSE: {:.0f}'.format(lasso_optimal_RMSE) + \
          ", R^2 = {:.3f}".format(optimal_lassoModel.score(XTest, yTest)))
plt.show()

#%%
multiple_lasso = np.vstack((multiple_regression_coefficients, lasso_coefficients)).T

bar_width = 0.25
fig = plt.figure(figsize = (14, 10))

br1 = np.arange(len(multiple_lasso))
br2 = [x + bar_width for x in br1]

plt.bar(br1, multiple_lasso[:, 0], width=bar_width, label="multiple OLS regression")
plt.bar(br2, multiple_lasso[:, 1], width=bar_width, label="lasso regression")

ticks = [r + bar_width for r in range(len(multiple_lasso))]
plt.xticks( ticks, features, rotation='vertical')

plt.title("Coefficients for Multiple vs. Lasso")

plt.legend()
plt.show()
#%% - NOTES
# - optimal lambda changes based on test_size and seed
# - attempt to find seed and test that results in lowest RMSE
# - chose 1009 cause prime number
# - 50-50 test split because?? lower RMSE than multiple regression with entire thing

#%%
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from scipy.special import expit

#%%
# Class Counts and Visualizations

y = tech_PD_regression['Gender (Encoded)']
x = tech_PD_regression['totalyearlycompensation']

# male - 17605, female - 3880
plt.bar([0,1], tech_PD_regression['Gender (Encoded)'].value_counts(ascending=True), color = ['b', 'r'], tick_label = ['Female', 'Male'])
f_percent = 3880 / (17605 + 3880) * 100
m_percent = 17605 / (17605 + 3880) * 100
plt.text(-0.2, 5000, '3880 - {:.2f}%'.format(f_percent))
plt.text(0.8, 17700, '17605 - {:.2f}%'.format(m_percent))
plt.ylabel('Count')
plt.title('Male/Female Class Counts');

plt.show()

compensation_coefficients = []
features_tested = []

#%%

# predict function to take in a particular threshold
def predict(probabilities, threshold):
    true_false = list(probabilities[:, 1] > threshold)
    array = []
    for ii in true_false:
        if ii:
            array.append(0)
        else:
            array.append(1)
    return np.array(array)

def logistic_regression(X, y, predictors, return_value):
    X = np.array(X)
    y = np.array(y)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=test_size, random_state=state)
    logisticModel = LogisticRegression(solver='liblinear', class_weight='balanced')
    logisticModel.fit(xTrain, yTrain)

    pred = logisticModel.predict(xTest)

    # See the percentage of examples that are correctly classified.
    accuracy = metrics.accuracy_score(yTest, pred) 
    print("Accuracy = {:0.1f}%".format(accuracy * 100))
    
    # Precision score.
    precision = metrics.precision_score(yTest, pred)
    print("Precision = {:0.1f}%".format(100 * precision))

    # Recall score.
    recall = metrics.recall_score(yTest, pred)
    print("Recall    = {:0.1f}".format(100 * recall))

    # F1 Score
    print("F1 Score   ={:0.3f}".format(f1[ix]))

    # MCC Score
    mcc = metrics.matthews_corrcoef(yTest, pred) 
    print("MCC    = {:0.3f}".format(mcc))

    #ROC-AUC Score
    roc_auc = metrics.roc_auc_score(yTest, pred)
    print("AUROC    = {:0.3f}".format(roc_auc))

    #PR Score
    precision = metrics.average_precision_score(yTest, pred)
    print("AUPRC    = {:0.3f}".format(precision))
    
    metrics_text = "Accuracy = {:0.1f}%".format(accuracy * 100) + "\n" + "Precision = {:0.1f}%".format(100 * precision) + \
        "\n" + "Recall    = {:0.1f}%".format(100 * recall) + "\n" + "AUROC    = {:0.3f}".format(roc_auc) + "\n" + \
            "AUPRC    = {:0.3f}".format(precision) + "\n" +  "F1 Score = {:0.3f}".format(f1[ix]) \
            + "\nMCC    = {:0.3f}".format(mcc)

    # The matrix of predictions and true values for each class.
    conf_matrix = metrics.confusion_matrix(yTest, pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels  = ['Male', 'Female'])

    fig, axs = plt.subplots(1)
    axs.set_title("Confusion matrix, Predictors: " + predictors)
    axs.text(-1, 2.5, metrics_text)
    cm_display.plot(ax=axs)
    plt.show()   
    
    if return_value == 'auroc':
        return roc_auc
    
    # coefficient of total annual compensation    
    return logisticModel.coef_[0][0]
    


#%% - NUMBER 4
# There is controversy as to the existence of a male/female gender pay gap in tech job
# compensation. Build a logistic regression model (with gender as the outcome variable)
# to see if there is an appreciable beta associated with total annual compensation with
# and without controlling for other factors.

# ONLY total_annual_compensation as predictor

# gender already encoded as categorical -> Male: 0, Female: 1
# because female is a minority should be the "positive" class -> so 0?
#y = np.array(y)
x = np.array(x)
y = np.array(y)

# weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0,1], y=y)
# weights = dict(zip([0,1], list(weights)))
xTrain, xTest, yTrain, yTest = train_test_split(x.reshape(-1,1), y, test_size=test_size, random_state=state)
logisticModel = LogisticRegression(solver='liblinear', class_weight='balanced')
logisticModel.fit(xTrain, yTrain)

probabilities = logisticModel.predict_proba(xTest)
# probabilities for positive class (Female)
positive_probabilities = probabilities[:, 1]

#pred = logisticModel.predict(xTest)

# find optimal thresholds for precision-recall (because class is imbalanced)
precision, recall, thresholds = precision_recall_curve(yTest, positive_probabilities)

# best balance of precision and recall -> f1 score
f1 = (2 * precision * recall) / (precision + recall)
ix = np.argmax(f1)
print("threshold =", thresholds[ix], "f1 score =", f1[ix])

female = len(yTest[yTest==1]) / len(yTest)

plt.plot([0, 1], [female, female], linestyle='--', label='Female')

plt.plot(recall, precision, marker='.', label='Logistic')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
plt.title("Precision-Recall Curve")
plt.xlabel('Recall (False Positive Rate)')
plt.ylabel('Precision (True Positive Rate)')
plt.legend()
plt.show()

# set optimal threshold to value at that index
optimal_threshold = round(thresholds[ix], 3)

pred = predict(probabilities, optimal_threshold)
#pred = logisticModel.predict(xTest)

# See the percentage of examples that are correctly classified.
accuracy = metrics.accuracy_score(yTest, pred) 
print("Accuracy = {:0.1f}%".format(accuracy * 100))

# The matrix of predictions and true values for each class.
conf_matrix = metrics.confusion_matrix(yTest, pred)

# Precision score.
precision = metrics.precision_score(yTest, pred)
print("Precision = {:0.1f}%".format(100 * precision))

# Recall score.
recall = metrics.recall_score(yTest, pred)
print("Recall    = {:0.1f}".format(100 * recall))

# F1 Score
print("F1 Score   ={:0.3f}".format(f1[ix]))

# MCC Score
mcc = metrics.matthews_corrcoef(yTest, pred) 
print("MCC    = {:0.3f}".format(mcc))

#ROC-AUC Score
roc_auc = metrics.roc_auc_score(yTest, pred)
print("AUROC    = {:0.3f}".format(roc_auc))

#PR Score
precision = metrics.average_precision_score(yTest, pred)
print("AUPRC    = {:0.3f}".format(precision))

metrics_text = "Accuracy = {:0.1f}%".format(accuracy * 100) + "\n" + "Precision = {:0.1f}%".format(100 * precision) + \
    "\n" + "Recall    = {:0.1f}%".format(100 * recall) + "\n" + "Threshold = {:0.3f}".format(optimal_threshold) + "\n" + \
    "AUROC    = {:0.3f}".format(roc_auc) + "\n" + "AUPRC    = {:0.3f}".format(precision) + "\n" +  "F1 Score = {:0.3f}".format(f1[ix]) \
        + "\nMCC    = {:0.3f}".format(mcc)
    
# as opposed to sklearn 0.5 threshold that had 81.9% accuracy but 0% precision and recall

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels  = ['Male', 'Female'])
fig, axs = plt.subplots(1)
axs.set_title("Confusion matrix for Annual Compensation as Predictor")
axs.text(-1, 2.5, metrics_text)
cm_display.plot(ax=axs)
plt.show()

print(logisticModel.coef_, logisticModel.intercept_)
# coef -2.64428255e-07, NOT APPRECIABLE

compensation_coefficients.append(logisticModel.coef_[0][0])
features_tested.append("Just Compensation")

#%%

logit_text = "Coefficients: " + str((logisticModel.coef_[0][0])) + "\n" + "Intercept: " + str(logisticModel.intercept_[0])
plt.clf()
x_lin = np.linspace(-int(20e6), int(20e6), int(40e6+1))
fig, axs = plt.subplots(1)
plt.scatter(x, y, color="black")#, zorder=20)
loss = expit(x_lin * logisticModel.coef_ + logisticModel.intercept_).ravel()
plt.plot(x_lin, loss, label="Logistic Regression Model", color="red")#, linewidth=3)
plt.title("Logit function for Annual Compensation & Gender")
axs.text(2000000, 0.7, logit_text)
plt.show()


#%%
fig, axs = plt.subplots(2, figsize=(10, 10))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels  = ['Male', 'Female'])
axs[0].set_title("Confusion matrix for Annual Compensation as Predictor")
axs[0].text(-2, 0.5, metrics_text)
cm_display.plot(ax=axs[0])

x_lin = np.linspace(-int(20e6), int(20e6), int(40e6+1))
axs[1].scatter(x, y, color="black")#, zorder=20)
loss = expit(x_lin * logisticModel.coef_ + logisticModel.intercept_).ravel()
axs[1].plot(x_lin, loss, label="Logistic Regression Model", color="red")#, linewidth=3)
axs[1].set_title("Logit function for Annual Compensation & Gender")
axs[1].text(2000000, 0.7, logit_text)


#%%
# NO manual threshold --> better for multiple

# TO-DO: Precision-Recall Scores?? Nah

y = tech_PD_regression['Gender (Encoded)']

# all features - NOPE
X = tech_PD_regression.drop('Gender (Encoded)', axis=1)
params = logistic_regression(X, y, "All Features", 0)
compensation_coefficients.append(params)
features_tested.append("All Features")
print(params)
# [array([[-5.35996582e-07, -1.72906175e-02, -1.04506020e-02,
#          2.31157117e-03, -7.29639026e-04, -7.78791211e-04,
#         -4.21710429e-04,  2.40373046e-04,  3.37783321e-04,
#          8.19106605e-04, -4.28059577e-04, -4.77944435e-04,
#          5.95327688e-03, -1.46173609e-03, -1.15137433e-04,
#          1.41447126e-04]]), array([8.45891533e-05])]

# first 3 columns (experience) - YES
X = tech_PD_regression.iloc[:, :3]
params = logistic_regression(X, y, "Compensation + Experience", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + Experience")
print(params)
# [array([[ 1.91488115e-07, -1.31443642e-02, -6.16307679e-03]]), array([0.00169113])]


# education + SAT + GPA - NOPE
X = tech_PD_regression[['totalyearlycompensation', 'Masters_Degree', 'Doctorate_Degree', 'Bachelors_Degree', 'SAT', 'GPA']]
params = logistic_regression(X, y, "Education and Test Scores", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + Education/Test Scores")
print(params)
# [array([[-9.92771262e-07,  1.34336526e-04, -4.18646296e-05,
#         -4.71772968e-05, -2.68127013e-05,  1.77576572e-04,
#          8.90262976e-06]]), array([5.60328876e-06])]

# race --> NOPE
X = tech_PD_regression[['totalyearlycompensation', 'Race_Asian', 'Race_White', 'Race_Black', 'Race_Hispanic']]
params = logistic_regression(X, y, "Race", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + Race")
print(params)

# height and age - NOPE
X = tech_PD_regression[['totalyearlycompensation', 'Height', 'Age']]
params = logistic_regression(X, y, "Height/Age", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + Height/Age")
print(params)
# [array([[-8.15984815e-07,  5.82028461e-03, -6.30296330e-03]]), array([8.66748161e-05])]

# first 2 - YES
X = tech_PD_regression[['totalyearlycompensation', 'yearsofexperience']]#, 'yearsatcompany', 'SAT', 'GPA']]
params = logistic_regression(X, y, "Compensation + YOE", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + YOE")
print(params)
# [array([[ 2.27022634e-07, -1.76423549e-02]]), array([0.0117241])]

# first 2 - NOPE VERY NOPE
X = tech_PD_regression[['totalyearlycompensation', 'yearsatcompany']]
params = logistic_regression(X, y, "Compensation + YAC", 0)
compensation_coefficients.append(params)
features_tested.append("Compensation + YAC")
print(params)
# [array([[-2.64428268e-07, -1.17583106e-11]]), array([9.64324695e-13])]

#%% NUMBER 4 - Plot
print(compensation_coefficients)
print(features_tested)

bar_width = 0.25
fig = plt.figure(figsize = (10, 10))

br1 = np.arange(len(compensation_coefficients))

plt.bar(br1, compensation_coefficients, width=bar_width)
ticks = [r for r in range(len(compensation_coefficients))]
plt.xticks( ticks, features_tested, rotation=90)

plt.title("Compensation Coefficients for different Controls")
plt.xlabel("Variables Controlled For")
plt.ylabel("Compensation Coefficient")

plt.legend()
plt.show()

#%% - NUMBER 5
# Build a logistic regression model to see if you can predict high and low pay from years
# of relevant experience, age, height, SAT score and GPA, respectively

auroc_vals = []
feature_vals = []

# less nan values since less columns to look at
tech_PD_logistic = tech_PD[['Age', 'Height', 'SAT', 'GPA', 'yearsofexperience']]
# no nulls
y = tech_PD['totalyearlycompensation']
median = y.median()

y = (y > median).map({True: 1, False:0})
X = tech_PD_logistic

for column in X:
    print(column)
    auroc = logistic_regression( np.array(X[column]).reshape(-1,1), y, column, 'auroc')
    auroc_vals.append(auroc)
    feature_vals.append(column)
    print()
    
# All Features
print("All features")
auroc = logistic_regression(X, y, "All features", "auroc")
auroc_vals.append(auroc)
feature_vals.append("All Features")

#%% 5 - Plot

print(auroc_vals)
print(feature_vals)

bar_width = 0.25
fig = plt.figure(figsize = (10, 10))

br1 = np.arange(len(auroc_vals))

plt.bar(br1, auroc_vals, width=bar_width)

y = np.linspace(0.6, 0.6, 20)
x = np.linspace(-0.1, 5.5, 20)

plt.plot(x, y, '--')
ticks = [r for r in range(len(auroc_vals))]
plt.xticks( ticks, feature_vals, rotation=90)

plt.title("Auroc for Different Features")
plt.xlabel("Feature")
plt.ylabel("AUROC")

plt.legend()
plt.show()

#%% - EXTRA CREDIT
# a) Is salary, height or age normally distributed? Does this surprise you? Why or why not?
tech_PD_extra = tech_PD[['Age', 'Height', 'totalyearlycompensation']]


for feature in tech_PD_extra.columns:
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    axs.hist(tech_PD_extra[feature], bins=20)
    plt.title(feature)


#%%
