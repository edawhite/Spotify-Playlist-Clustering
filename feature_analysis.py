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
