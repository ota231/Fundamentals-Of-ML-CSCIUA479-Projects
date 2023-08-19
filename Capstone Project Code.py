# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:07:36 2023

@author: tomis
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize


#%%
from umap import UMAP
import umap.plot
#%%
net_id = 12670299
random.seed(net_id)
np.random.seed(net_id)
#%%
songs = pd.read_csv(r"C:\Users\ota231\Downloads\musicData.csv")

# 5 completely nan values (including genre), drop them and reset the indices
songs = songs.drop(np.where(songs['music_genre'].isnull())[0])
songs = songs.reset_index(drop=True)

#%%
genres = list(set(songs['music_genre']))
# extract genre based data
alternative_df = songs.iloc[np.where(songs['music_genre'] == 'Alternative')[0], :]
country_df = songs.iloc[np.where(songs['music_genre'] == 'Country')[0], :]
anime_df =  songs.iloc[np.where(songs['music_genre'] == 'Anime')[0], :]
rock_df = songs.iloc[np.where(songs['music_genre'] == 'Rock')[0], :]
hip_hop_df = songs.iloc[np.where(songs['music_genre'] == 'Hip-Hop')[0], :]
jazz_df = songs.iloc[np.where(songs['music_genre'] == 'Jazz')[0], :]
rap_df = songs.iloc[np.where(songs['music_genre'] == 'Rap')[0], :]
blues_df = songs.iloc[np.where(songs['music_genre'] == 'Blues')[0], :]
classical_df =  songs.iloc[np.where(songs['music_genre'] == 'Classical')[0], :]
electronic_df = songs.iloc[np.where(songs['music_genre'] == 'Electronic')[0], :]

genres_dfs = [alternative_df, country_df, anime_df, rock_df, hip_hop_df, jazz_df, rap_df, blues_df, \
              classical_df, electronic_df]

#%%
# split randomly into train an test (10%) sets
def get_train_test(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size = 0.1, random_state=net_id)
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    return train, test

def get_train_test_dfs(genres_dfs):
    train_list = []
    test_list = []
    for df in genres_dfs:
        train, test = get_train_test(df)
        train_list.append(train)
        test_list.append(test)
    return train_list, test_list

#%% 
train_list, test_list = get_train_test_dfs(genres_dfs)

#%%

# use linear regression to impute values
def regression_imputation(df, columns):
    imputed_df = df.copy()

    # only rows with missing values in columns
    missing_values_df = imputed_df.loc[imputed_df[columns].isnull().any(axis=1)]

    # only rows w/o missing values in columns
    non_missing_values_df = imputed_df.loc[~imputed_df[columns].isnull().any(axis=1)]

    for col in columns:
        X = non_missing_values_df.drop(columns=columns + [col])
        y = non_missing_values_df[col]
        reg_model = LinearRegression().fit(X, y)

        missing_values = reg_model.predict(missing_values_df.drop(columns=columns + [col]))
        imputed_df.loc[missing_values_df.index, col] = missing_values

    return imputed_df

# takes in a genre df and returns it preprocessed (nan values removed, etc)
def preprocess_genre(df):
    df = df.copy()
    # convert "missing" values to nans: duration(-1), instrumentalness(0), tempo(?)
    df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)
    df['instrumentalness'] = df['instrumentalness'].replace(0, np.nan) 
    df['tempo'] = df['tempo'].replace('?', np.nan) 

    # make them appropriate data tyoes
    df['duration_ms'] = df['duration_ms'].astype(float)
    df['instrumentalness'] = df['instrumentalness'].astype(float)
    df['tempo'] = df['tempo'].astype(float)
    
    genre = df['music_genre']
    
    # get columns useful for prediction
    useful_columns = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
      'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']

    # label encode key, binarize mode
    df = df[useful_columns]
    df['mode'] = df['mode'].replace({'Major': 1, 'Minor': 0})

    le = LabelEncoder()
    df['key'] = le.fit_transform(df['key'])
    
    imputed_df = regression_imputation(df, ['duration_ms', 'tempo', 'instrumentalness'])
    imputed_df['music_genre'] = genre
    
    return imputed_df

def rescale(df):
    df = df.reset_index(drop=True)
    
    original_columns = list(df.columns)
    
    df_to_scale = df.drop(['mode', 'key', 'music_genre'], axis=1)
    scaled_columns = list(df_to_scale.columns)
    
    sc = StandardScaler()
    scaled = sc.fit_transform(df_to_scale)
    
    scaled_df = pd.DataFrame(scaled, columns=scaled_columns)
        
    original_df = pd.concat([scaled_df, df[['mode', 'key', 'music_genre']]], axis=1)
    original_df = original_df[original_columns]
    
    return original_df


#%%
preprocessed_train = []
for df in train_list:
    preprocessed_train.append(preprocess_genre(df))
    
#%%
scaled = []
for df in preprocessed_train:
    scaled.append(rescale(df))

#%%
train = pd.concat(scaled)
train = train.reset_index(drop=True)

#%%
mapper = UMAP(min_dist=0.5, n_neighbors=100).fit(train.drop('music_genre', axis=1).values)
umap.plot.points(mapper, labels=train['music_genre'])

#%%
t_sne = TSNE(n_components=2, learning_rate='auto', perplexity=20, n_jobs=-1)
t_sne_embedding = t_sne.fit_transform(train.drop('music_genre', axis=1).values)

#%%
colors = ['#F94144', '#F3722C', '#F8961E', '#F9C74F', '#90BE6D', '#43AA8B', '#4D908E', '#577590', '#277DA1', '#6D597A']
#%%
actual_labels = train['music_genre']
for i, label in enumerate(set(actual_labels)):
    x = t_sne_embedding[actual_labels == label, 0]
    y = t_sne_embedding[actual_labels == label, 1]
    plt.scatter(x, y, color=colors[i % len(colors)], label=label,  marker='.', alpha=0.5, s=0.4)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=20)
plt.title("T-sne embedding of data, KL-Divergence={:.2f}".format(t_sne.kl_divergence_))
plt.show()

#%%
kmeans = KMeans(10, n_init='auto')
kmeans.fit(t_sne_embedding)
labels = kmeans.fit_predict(t_sne_embedding)

#%%
plot_labels = np.unique(labels)
  
for i in plot_labels:
    plt.scatter(t_sne_embedding[labels == i , 0] , t_sne_embedding[labels == i , 1] , label = "Class " + str(i+1), marker='.')
plt.legend()
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Kmeans Clustering (t-SNE)")
plt.show()

#%% show not optimal by silhouette sum

k_values = np.linspace(2,12,11)
k_silhouete_df = pd.DataFrame(columns=k_values, index=['Silhouette Sum'])

for k in k_values:
    kmeans = KMeans(int(k), n_init='auto')
    labels = kmeans.fit_predict(t_sne_embedding)
    silhouette_scores = silhouette_samples(t_sne_embedding, labels)
    sum_silhouette_scores = sum(silhouette_scores)
    k_silhouete_df[k] = sum_silhouette_scores
    
#%%
# plot silhouette sum graph
plt.plot(k_values, k_silhouete_df.T['Silhouette Sum'])
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Silhouette Scores")
plt.xticks(k_values)

#%%
preprocessed_test = []
for df in test_list:
    preprocessed_test.append(preprocess_genre(df))
    
#%%
scaled = []
for df in preprocessed_test:
    scaled.append(rescale(df))
    
# auc higher without scaled values    

#%%
test = pd.concat(preprocessed_test)
test = test.reset_index(drop=True)

train = pd.concat(preprocessed_train)
train = train.reset_index(drop=True)

#%%
# testing - imputing vs. not imputing,
X_train, X_test = train.drop('music_genre', axis=1).values, test.drop('music_genre', axis=1).values

le = LabelEncoder()

y_train, y_test = le.fit_transform(train['music_genre']), le.fit_transform(test['music_genre'])
#%%

model = RandomForestClassifier()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)

#%%
print("macro", roc_auc_score(y_test, probs, multi_class='ovr'))
print("micro", roc_auc_score(y_test, probs, multi_class='ovr', average='micro'))

#%%
# plot auc curvessss
# binarize the labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# calculate fpr, tpr and auc for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs.ravel())
roc_auc["micro"] = roc_auc_score(y_test_bin, probs, average="micro")

# compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 10
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         linestyle='--', linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         linestyle='--', linewidth=2)

for i in range(10):
    plt.plot(fpr[i], tpr[i], label=genres[i]+' (AUC = %0.2f)' % roc_auc[i])


plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curves')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#%% Extra credit - predict duration
plt.hist(train['duration_ms'] / 60000, bins=50)
plt.title("Distribution of duration (in minutes) in train set")

#%% median SPLITS

train_med = (train['duration_ms'] / 60000).median()
test_med = (test['duration_ms'] / 60000).median()

train['duration_med'] = (train['duration_ms'] / 60000).apply(lambda x: 1 if x >= train_med else 0)
test['duration_med'] = (test['duration_ms'] / 60000).apply(lambda x: 1 if x >= test_med else 0)


#%%

train['music_genre'], test['music_genre'] = le.fit_transform(train['music_genre']), le.fit_transform(test['music_genre'])
ref_cols = train.drop(['duration_ms', 'duration_med'], axis=1).columns
X_train, X_test = train.drop(['duration_ms', 'duration_med'], axis=1).values, test.drop(['duration_ms', 'duration_med'], axis=1).values

y_train, y_test = train['duration_med'], test['duration_med']

#%%
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

#%%
extra_auc = roc_auc_score(y_test, probs[:, 1])
coefs = model.coef_
# array([[-6.75169103e-03, -6.45848863e-01, -2.53213664e+00,
#          8.69717766e-01,  4.04907359e-01,  1.87723431e-03,
#          1.72081052e-01, -1.34280018e-02, -1.33020832e-01,
#         -4.10727006e+00, -2.04966852e-03, -1.06045032e+00,
#          2.08392626e-02]])
#%%
fig, ax = plt.subplots()
ax.bar(ref_cols, coefs[0])

ax.set_title('Model Coefficients for each feature')
ax.set_xlabel('Features')
ax.set_ylabel('Coefficients')
plt.xticks(rotation=90, ha='right')

plt.show()


#%%