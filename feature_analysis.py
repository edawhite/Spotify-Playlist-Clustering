import pandas as pd
from tabulate import tabulate
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn import preprocessing
from sklearn.cluster import KMeans
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl
from yellowbrick.cluster import KElbowVisualizer

df = pd.read_csv("hard_bop_jazz.csv")
df1 = pd.read_csv("hard_bop_jazz.csv")
df1.drop(['Artist Name(s)','Duration (ms)', 'Artist IDs','Spotify ID', 'Genres', 'Mode', 'Key'], axis=1, inplace=True)

#  ------------------------ Finding k  -----------------------------

#Picking the parameters to cluster around
df1 = df[['Instrumentalness', 'Energy', 'Loudness']]

x = df1.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled2 = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x_scaled2)


#  ------------------------ k-means Clustering ---------------------

kmeans = KMeans(init="k-means++",
                                n_clusters=3,
                                random_state=0,
                                max_iter = 500).fit(x_scaled2)
df1['kmeans'] = kmeans.labels_
df1.columns = ['Energy', 'Instrumentalness', 'Loudness','kmeans' ]


kmeans = df1['kmeans']
df['kmeans'] = kmeans

fig = px.scatter_3d(df, x='Energy', y='Instrumentalness', z='Loudness',
                                  color='kmeans', hover_name='Track Name')
fig.show()
#3D scatter plot of k-means clustering with names of tracks included by hovering over each data pt

# ---------------------------------- Violin Plot for each cluster ------------------
#c0 = df1[df1['kmeans']==0]
#c1 = df1[df1['kmeans']==1]
#c2 = df1[df1['kmeans']==2]
#c3 = df1[df1['kmeans']==3]


# genre =df ['genre']
# c0['genre'] = genre
# c1['genre'] = genre
# c2['genre'] = genre

#c0.drop(['kmeans'], axis=1, inplace=True)
#c1.drop(['kmeans'], axis=1, inplace=True)
#c2.drop(['kmeans'], axis=1, inplace=True)
#c3.drop(['kmeans'], axis=1, inplace=True)



#x = c0.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#c0_scaled = min_max_scaler.fit_transform(x)
#c0 = pd.DataFrame(c0_scaled)
#c0.columns = ['Energy', 'Instrumentalness', 'Loudness' ]
#c0=c0.melt(var_name='groups', value_name='vals')

#x = c1.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#c1_scaled = min_max_scaler.fit_transform(x)
#c1 = pd.DataFrame(c1_scaled)
#c1.columns = ['Energy', 'Instrumentalness', 'Loudness' ]
#c1=c1.melt(var_name='groups', value_name='vals')

#x = c2.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#c2_scaled = min_max_scaler.fit_transform(x)
#c2 = pd.DataFrame(c2_scaled)
#c2.columns = ['Energy', 'Instrumentalness', 'Loudness']
#c2=c2.melt(var_name='groups', value_name='vals')

#x = c3.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#c3_scaled = min_max_scaler.fit_transform(x)
#c3 = pd.DataFrame(c3_scaled)
#c3.columns = ['Energy', 'Instrumentalness', 'Loudness']
#c3=c3.melt(var_name='groups', value_name='vals')

#f, axes = plt.subplots(4, 1)
#ax = sns.violinplot( data=c0 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
#ax = sns.violinplot( data=c1 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
#ax = sns.violinplot( data=c2 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
#ax = sns.violinplot( data=c3 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])

#plt.show()
