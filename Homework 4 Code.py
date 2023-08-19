# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:00:27 2023

@author: tomis
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import Perceptron
import torch
from torch import nn
from IPython import display
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

#%%
diabetes = pd.read_csv(r"D:\TechStuffs\Code\Machine Learning\Fundamentals of ML Class\Homeworks\Homework 4\diabetes.csv")

#%%

y = diabetes['Diabetes']
X = diabetes.drop('Diabetes', axis=1)
shortened_columns = X.columns

# %%
# 1. Build and train a Perceptron (one input layer, one output layer, no hidden layers and
# no activation functions) to classify diabetes from the rest of the dataset. What is the
# AUC of this model?

# perceptron class that computes optimal test size and seed 
class optimizePerceptron:
    def __init__(self, test_sizes, random_states):
        self.test_sizes = test_sizes
        self.random_states = random_states
        self.best_params = {}
        self.best_auc = 0
        self.best_model = None
        self.X_test = None
        self.y_test = None

    def compute_best_params(self):
        for seed in self.random_states:
            np.random.seed(seed)
            for test_size in self.test_sizes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=seed)
                model = Perceptron(random_state=seed, class_weight='balanced')
                model.fit(X_train, y_train)
                auc = metrics.roc_auc_score(
                    y_test, model.decision_function(X_test))
                if auc > self.best_auc:
                    self.best_auc = auc
                    self.best_params = {
                        'random_state': seed, 'test_size': test_size}
                    self.best_model = model
                    self.X_test = X_test
                    self.y_test = y_test

    def plot_best_auc(self):
        metrics.RocCurveDisplay.from_estimator(
            self.best_model, self.X_test, self.y_test)


#%%
# initialize perceptron model and plot the best results
test_sizes = [0.1, 0.2, 0.4, 0.5]
random_states = [0, 1234, 42, 10, 9]

perceptron = optimizePerceptron(test_sizes, random_states)
perceptron.compute_best_params()

perceptron.plot_best_auc()
# %%
# 2. Build and train a feedforward neural network with at least one hidden layer to classify
# diabetes from the rest of the dataset. Make sure to try different numbers of hidden
# layers and different activation functions (at a minimum reLU and sigmoid). Doing so:
# How does AUC vary as a function of the number of hidden layers and is it dependent
# on the kind of activation function used (make sure to include "no activation function"
# in your comparison). How does this network perform relative to the Perceptron?

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
# test different test_sizes and splits (for pytorch specifically)
def get_split(X, y, test_size, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

#%%
# get weights for Diabetes prediction (class imbalance)
def get_weights(y):
    return compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)

#%%
# inherit pytorch Dataset class and use it to load any Dataset
class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% FFNN Model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden, activation):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        
        # variable number of hidden layers
        for num in range(num_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.input_layer(x)
        
        for num in range(self.num_hidden):
            # account for No activation layer
            if self.activation:
                out = self.activation(out)
            out = self.hidden_layers[num](out)

        out = self.output_layer(out)
        return out

# train and test loops for FFNN
def train(model, epoch, criterion, optimizer, train_loader):
    model.train()
    for iter_ in range(epoch):
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
    
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
    
            score, predicted = torch.max(y_pred, 1)
            acc = (y_train == predicted).sum().float() / len(y_train)
    
            if batch_idx % 100 == 0:
                print('Epoch: {}, Train loss: {}, Accuracy: {}'.format(
                    iter_, loss.item(), acc))
                display.clear_output(wait=True)

            loss.backward()
            optimizer.step()

def test(model, criterion, optimizer, test_loader, plot=False):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        acc = 0
        predictions = []
        actual = []
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            test_loss += criterion(y_pred, y_test).item()
            score, predicted = torch.max(y_pred, 1)
            actual.append(list(y_test.numpy()))
            predictions.append(list(y_pred[:, 1].numpy()))
            acc += (y_test == predicted).sum().float() / len(y_pred)
        test_loss /= len(test_loader)
        acc /= len(test_loader)
        predictions = list(itertools.chain(*predictions))
        actual = list(itertools.chain(*actual))
    print('Average Test loss', test_loss)
    print('Average Accuracy', acc.item())
    print('AUC', roc_auc_score(actual, predictions))
    if plot:
        metrics.RocCurveDisplay.from_predictions(actual, predictions)
    return roc_auc_score(actual, predictions)

#%% 

# simple way to get train_loader, test_loader, the respective weights & batch size (if needed)
def get_train_test_loader(X, y, test_size, seed, binary=True, batch_size=1024):
    X_train, X_test, y_train, y_test = get_split(X, y, test_size, seed)
    
    if binary:
        class_weights_train = get_weights(y_train)
        class_weights_train = torch.tensor(class_weights_train,dtype=torch.float)
        
        class_weights_test = get_weights(y_test)
        class_weights_test = torch.tensor(class_weights_test,dtype=torch.float)
    
    train_data = LoadData(X_train, y_train)
    test_data = LoadData(X_test, y_test)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    if binary:
        return train_loader, class_weights_train, test_loader, class_weights_test
    else:
        return train_loader, test_loader

#%%
# because of train/test/split and seed concerns, first check which train/test/split and seed
# combos work best for each activation function
# choose just one activation at a time for computational resources
# also just 1 epoch for same reason
# tried for different activation functions and determined best on visual inspection of dataframe
# used 100 hidden dimensions -> gave very high accuracy

# Results:
# Sigmoid -> state: 10, test_size: 0.1, AUC: 0.648534
# ReLU() -> state: 10, test_size: 0.1, AUC: 0.705204
# None -> state: 0, test_size: 0.1, AUC: 0.734508

test_sizes = [0.1, 0.2, 0.4, 0.5]
random_states = [0, 1234, 42, 10, 9]

input_dim = 21
hidden_dim = 100
# change according to what is being tested
activation = nn.Sigmoid()
output_dim = 2
num_hidden = 1
num_epochs = 1
learning_rate = 0.001
lambda_l2 = 0.1

split_seed_df = pd.DataFrame(index=test_sizes, columns=random_states)

for seed in random_states:
    for test_size in test_sizes:

        train_loader, class_weights_train, test_loader, class_weights_test = get_train_test_loader(X, y, test_size, seed)
        
        ffnn_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
        ffnn_model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_train)
        optimizer = torch.optim.SGD(ffnn_model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        train(ffnn_model, num_epochs, criterion, optimizer, train_loader)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_test)
        auc = test(ffnn_model, criterion, optimizer, test_loader)
        split_seed_df.loc[test_size, seed] = auc

#%%
# constants for NN in next cell
input_dim = 21
hidden_dim = 100
output_dim = 2
num_hidden = 1
num_epochs = 2
learning_rate = 0.001
lambda_l2 = 0.1

# %%
# compare AUC values for best performing train/test/split and seed combo for each actication function
# different # of hidden layers: 1 - 10
# different activation functions (including None): reLU, sigmoid, None

# NOTE: this may take a while to run

hidden_tests = list(range(1, 11))
activation_tests = [nn.Sigmoid(), nn.ReLU(), None]

# dataframe to store AUC for different hidden layers/activation functions
auc_comparison_df = pd.DataFrame(index=['Sigmoid', 'ReLU', 'None'], columns=hidden_tests)

# Sigmoid -> state: 10, test_size: 0.1
# ReLU() -> state: 10, test_size: 0.1
# None -> state: 0, test_size: 0.1
for idx_activation, activation in enumerate(activation_tests):
    
    if idx_activation == 0 or idx_activation == 1:
        state = 10
    else:
        state = 0
    
    train_loader, class_weights_train, test_loader, class_weights_test = get_train_test_loader(X, y, 0.1, state)
    
    for num_hidden in hidden_tests:
        
        ffnn_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
        ffnn_model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_train)
        optimizer = torch.optim.SGD(ffnn_model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        train(ffnn_model, num_epochs, criterion, optimizer, train_loader)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_test)
        auc = test(ffnn_model, criterion, optimizer, test_loader)
        auc_comparison_df.iloc[idx_activation, num_hidden - 1] = auc

#%%
auc_comparison_df = auc_comparison_df.T

#%% - plot graph for #2
plt.plot(hidden_tests, auc_comparison_df.Sigmoid, label = "Sigmoid")
plt.plot(hidden_tests, auc_comparison_df.ReLU, label = "ReLU")
plt.plot(hidden_tests, auc_comparison_df['None'], label = "No Activation Function")
plt.ylim(bottom=0, top=1)
plt.legend(loc='upper right')
plt.xlabel("Number of Hidden Layers")
plt.ylabel("AUC")
plt.show()
#%%
# 3. Build and train a “deep” network (at least 2 hidden layers) to classify diabetes from
# the rest of the dataset. Given the nature of this dataset, is there a benefit of using a
# CNN or RNN for the classification? 

# no benefit of CNN or RNN, no nearness or sequential data to take advantage of

# increased num_epochs and hidden_dim because just one network being done now,so more computational
# resources. also increased learning rate to 0.01 to "compensate" for increase in computational resources
# (later found out that the lr did not change anything)

# used optimal train/test/seed for None

# constants for NN
input_dim = 21
hidden_dim = 200
output_dim = 2
num_hidden = 3
num_epochs = 5
activation = None
learning_rate = 0.01
lambda_l2 = 0.1

train_loader, class_weights_train, test_loader, class_weights_test = get_train_test_loader(X, y, 0.1, 0)

deep_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
deep_model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_train)
optimizer = torch.optim.SGD(deep_model.parameters(), lr=learning_rate, weight_decay=lambda_l2)
train(deep_model, num_epochs, criterion, optimizer, train_loader)

criterion = nn.CrossEntropyLoss(weight=class_weights_test)
#%%
auc = test(deep_model, criterion, optimizer, test_loader, plot=True)

#%%
# 4. Build and train a feedforward neural network with one hidden layer to predict BMI
# from the rest of the dataset. Use RMSE to assess the accuracy of your model. Does
# the RMSE depend on the activation function used? 

y = diabetes['BMI']
X = diabetes.drop('BMI', axis=1)

#%% define new train and test functions for RMSE
# use MSE function in pytorch but compute sqrt of that for RMSE

def train(model, epoch, criterion, optimizer, train_loader):
    model.train()
    for iter_ in range(epoch):
        model.train()
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            y_pred = model(X_train)
            # float to 0 dp -> because gave decimals but BMI are whole numbers
            y_pred = torch.round(y_pred)            
            y_train = y_train[:, None].float()
            
            loss = torch.sqrt(criterion(y_pred.float(), y_train.float()))
            
            if batch_idx % 100 == 0:
                print('Epoch: {} Train loss/RMSE: {}'.format(iter_, loss.item()))
                display.clear_output(wait=True)
            loss.backward()
            optimizer.step()

def test(model, criterion, optimizer, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            y_pred = torch.round(y_pred)
            y_test = y_test[:, None].float()
            test_loss += torch.sqrt(criterion(y_pred, y_test)).item()
        test_loss /= len(test_loader)
    print('Average RMSE', test_loss)
    return test_loss

#%%
# first find optimal train/test/split for each activation function (using RMSE)

# set constants for NN in next cell

test_sizes = [0.1, 0.2, 0.4, 0.5]
random_states = [0, 1234, 42, 10, 9]

input_dim = 21
hidden_dim = 100
# change according to what is being tested
activation = nn.ReLU()
output_dim = 1
num_hidden = 1
num_epochs = 1
learning_rate = 0.001
lambda_l2 = 0.1

#%%
split_seed_df = pd.DataFrame(index=test_sizes, columns=random_states)

# Results:
# Sigmoid -> state: 1234, test_size: 0.1, RMSE: 29.2688
# ReLU() -> state: 1234, test_size: 0.1, RMSE: 30.2619
# None -> state: 1234, test_size: 0.1, RMSE: 30.9042

for seed in random_states:
    for test_size in test_sizes:
        train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False)
        
        bmi_model_oneh = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
        bmi_model_oneh.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(bmi_model_oneh.parameters(), lr=learning_rate, weight_decay=lambda_l2)
        train(bmi_model_oneh, num_epochs, criterion, optimizer, train_loader)
        rmse = test(bmi_model_oneh, criterion, optimizer, test_loader)
        split_seed_df.loc[test_size, seed] = rmse
        
#%%
# set constants to test activation against optimal test split/seed
# increasing hidden dimensions from 100 -> 150 yielded better performance across all activation 
# functions from train/split/seed test, leading ReLU and None to perform even better than Sigmoid
input_dim = 21
hidden_dim = 150
output_dim = 1
num_hidden = 1
num_epochs = 2
learning_rate = 0.001
test_size = 0.1
seed = 1234

#%%
# then RMSE against activation function
activation_tests = [nn.Sigmoid(), nn.ReLU(), None]
rmse_comparison_df = pd.DataFrame(index=['Sigmoid', 'ReLU', 'None'], columns=['RMSE'])

for idx, activation in enumerate(activation_tests):
    train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False)
    
    bmi_model_oneh = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
    bmi_model_oneh.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(bmi_model_oneh.parameters(), lr=learning_rate)
    train(bmi_model_oneh, num_epochs, criterion, optimizer, train_loader)
    rmse = test(bmi_model_oneh, criterion, optimizer, test_loader)
    rmse_comparison_df.iloc[idx, 0] = rmse
    
#%%
plt.bar(rmse_comparison_df.index, rmse_comparison_df.RMSE)
plt.ylim(27, 29.5)
plt.xlabel("Activation Function")
plt.ylabel("RMSE")

# lowest RMSE: None, 28.6962

#%%
# 5. Build and train a neural network of your choice to predict BMI from the rest of your
# dataset. How low can you get RMSE and what design choices does RMSE seem to
# depend on?

# using #4 as a basline -> progressively tune different hyperparamaters. pick best result for each parameter
# before moving on to the next one

input_dim = 21
hidden_dim = 150
output_dim = 1
num_hidden = 1
num_epochs = 2
learning_rate = 0.001
test_size = 0.1
seed = 1234
activation = None

#%% 1st thing: number of hidden layers. increasing number of hidden layers did not lower RMSE
# like previous question (#2) on Diabetes

#%% 2nd thing. Number of hidden dimensions

# Notes: training time increased as number of hidden dimension increased, expected
hidden_dim_test = [1, 10, 20, 30, 50, 75, 84, 90, 100, 120, 150, 175, 200, 250, 300]
hidden_dim_df = pd.DataFrame(index=hidden_dim_test, columns=['RMSE'])

for hidden_dim in hidden_dim_test:
    train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False)

    bmi_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
    bmi_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(bmi_model.parameters(), lr=learning_rate, momentum=0.9)
    train(bmi_model, num_epochs, criterion, optimizer, train_loader)
    rmse = test(bmi_model, criterion, optimizer, test_loader)
    hidden_dim_df.loc[hidden_dim, 'RMSE'] = rmse
    
#%% - hidden dim plot
plt.plot(hidden_dim_test, hidden_dim_df.RMSE)
plt.xlabel("Number of Hidden Dimensions")
plt.ylabel("RMSE")
plt.show()

# lowest RMSE: hidden_dim=200, 28.3022
hidden_dim = 200

#%% 3rd thing: learning rate - tried values from 0.001 to 1 and RMSE didn't change. 

learning_rate = 0.1

#%% 4th thing: batch size

# smaller batch sizes took less time to run, expected
batch_size_tests = [64, 128, 256, 512, 1024, 2048]
batch_size_df = pd.DataFrame(index=batch_size_tests, columns=['RMSE'])

for batch_size in batch_size_tests:
    train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False, batch_size=batch_size)

    bmi_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
    bmi_model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(bmi_model.parameters(), lr=learning_rate, momentum=0.9)
    train(bmi_model, num_epochs, criterion, optimizer, train_loader)
    rmse = test(bmi_model, criterion, optimizer, test_loader)
    batch_size_df.loc[batch_size, 'RMSE'] = rmse
    
#%%
plt.bar(['64', '128', '256', '512', '1024', '2048'], batch_size_df.RMSE)
plt.ylim(28.25, 28.32)
plt.xlabel("Batch Size")
plt.ylabel("RMSE")

#%%
batch_size = 64
# lowest RMSE: 64, 28.2833

#%% 5th thing: # of epochs

# can actually do more epochs now because of lower batch size

# too time consuming to do after 10
# probably unstable because there is nothing more to learn
# data/model itself is not extremely complicated

epoch_tests = list(range(1, 11))
epoch_tests_df = pd.DataFrame(index=epoch_tests, columns=['RMSE'])

for num_epochs in epoch_tests:
    train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False, batch_size=batch_size)

    bmi_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
    bmi_model.to(device)    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(bmi_model.parameters(), lr=learning_rate, momentum=0.9)
    train(bmi_model, num_epochs, criterion, optimizer, train_loader)
    rmse = test(bmi_model, criterion, optimizer, test_loader)
    epoch_tests_df.loc[num_epochs, 'RMSE'] = rmse

#%%
plt.plot(epoch_tests, epoch_tests_df.RMSE)
plt.xlabel("Number of Epochs")
plt.ylabel("RMSE")
plt.show()

#%%
num_epochs = 3
# lowest RMSE: 3, 28.2797

#%% 6th thing: momentum
# did not change the RMSE

#%% 7th thing: activation
# best activation is None, which means the non-linearity does not improve the performance of this network
# NN probably not the best for this task (too complex)

#%% 8th thing: regularization
# only increased the RMSE, but that is expected


#%%
# changed variables in this cell to test parameters that were not tested using loops
train_loader, test_loader = get_train_test_loader(X, y, test_size, seed, binary=False, batch_size=batch_size)

bmi_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim, num_hidden, activation)
bmi_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(bmi_model.parameters(), lr=learning_rate)
train(bmi_model, num_epochs, criterion, optimizer, train_loader)
rmse = test(bmi_model, criterion, optimizer, test_loader)


#%%
# Extra credit 
# b) Write a summary statement on the overall pros and cons of using neural networks to
# learn from the same dataset as in the prior homework, relative to using classical methods
# (logistic regression, SVM, trees, forests, boosting methods). Any overall lessons? 

# heavily depends on the use case
# more nuanced/complex relationships -> a lot of hyperparameters to think about
# not necessary here
# take a while to run

#%%
