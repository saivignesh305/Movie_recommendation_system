# import required libraries
import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# start measuring the time
start_time = time.time()
# load the datasets
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# combine relevant features
movies_df['cast'] = credits_df['cast'].apply(lambda x: ' '.join([i['name'] for i in eval(x)[:5]]))
movies_df['crew'] = credits_df['crew'].apply(lambda x: ' '.join([i['name'] for i in eval(x) if i['job'] == 'Director']))

# convert the text from relevant feature into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'] + ' ' + movies_df['cast'] + ' ' + movies_df['crew'])

# calculate the similarity between the vectors using cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# calculate the similarity between the vectors using Euclidean distance
euclidean_dist = 1 - cosine_sim

# create a kNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# create the feature matrix for kNN
X = tfidf_matrix.toarray()

# create the label vector for kNN
y = movies_df['title']

# fit the kNN classifier on the feature matrix and label vector
knn.fit(X, y)

# clustering the movies
num_clusters = 5 # choose the number of clusters
km = KMeans(n_clusters=num_clusters, random_state=0)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# add the cluster label to the movies dataframe
movies_df['cluster'] = clusters

# calculate the similarity between the vectors within each cluster
cluster_cosine_sim = {}
for i in range(num_clusters):
    cluster_indices = movies_df[movies_df['cluster'] == i].index.tolist()
    cluster_matrix = tfidf_matrix[cluster_indices]
    cosine_sim = cosine_similarity(cluster_matrix, cluster_matrix)
    cluster_cosine_sim[i] = cosine_sim

# get movie recommendations based on the given movie within the same cluster
def get_recommendations(title, cosine_sim, movies_df):
    cluster_label = movies_df[movies_df['title'] == title]['cluster'].values[0]
    cluster_indices = movies_df[movies_df['cluster'] == cluster_label].index.tolist()
    sim_scores = [(i, cosine_sim[cluster_label][idx]) for i, idx in enumerate(cluster_indices) if idx != indices[title]]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:10]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]



# get movie recommendations based on the given movie using kNN with Euclidean distance
def get_recommendations_knn(title, knn, tfidf, movies_df):
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    idx = indices[title]
    movie_vec = tfidf.transform([movies_df.loc[idx, 'genres'] + ' ' + movies_df.loc[idx, 'cast'] + ' ' + movies_df.loc[idx, 'crew']])
    _, movie_indices = knn.kneighbors(movie_vec.toarray())
    return movies_df['title'].iloc[movie_indices[0][1:]]

# example: get 10 movie recommendations based on 'The Dark Knight Rises' using cosine similarity
print("Using Content-Based Filtering")
print(get_recommendations('The Dark Knight Rises', cosine_sim, movies_df))

# example: get 10 movie recommendations based on 'The Dark Knight Rises' using kNN with Euclidean distance
print("Using KNN Algorithm")
print(get_recommendations_knn('The Dark Knight Rises', knn, tfidf, movies_df))

# end measuring the time
end_time = time.time()

# calculate the execution time
execution_time = end_time - start_time

# print the execution time
print("Execution time:", execution_time)


# calculate the score
if execution_time < 1:
    score = 100
elif execution_time < 2:
    score = 90
elif execution_time < 3:
    score = 80
elif execution_time < 4:
    score = 70
elif execution_time < 5:
    score = 60
elif execution_time < 6:
    score = 50
elif execution_time < 7:
    score = 40
elif execution_time < 8:
    score = 30
elif execution_time < 9:
    score = 20
else:
    score = 10
print("Performance Score", score)
# plot graphs of Runtime versus Number of movies, Runtime versus Budget, Runtime versus Number of movies
plt.scatter(movies_df['runtime'], movies_df['vote_count'])
plt.xlabel('Runtime')
plt.ylabel('Number of votes')
plt.show()

plt.scatter(movies_df['runtime'], movies_df['budget'])
plt.xlabel('Runtime')
plt.ylabel('Budget')
plt.show()

plt.scatter(movies_df['runtime'], movies_df['revenue'])
plt.xlabel('Runtime')
plt.ylabel('Revenue')
plt.show()

# plot bar graphs of Top Genres, Actors with highest appearance and Directors with highest movies
plt.barh(movies_df['genres'].value_counts().index[:10], movies_df['genres'].value_counts()[:10])
plt.xlabel('Number of movies')
plt.ylabel('Genres')
plt.show()

plt.barh(movies_df['cast'].value_counts().index[:10], movies_df['cast'].value_counts()[:10])
plt.xlabel('Number of movies')
plt.ylabel('Actors')
plt.show()

plt.barh(movies_df['crew'].value_counts().index[:10], movies_df['crew'].value_counts()[:10])
plt.xlabel('Number of movies')
plt.ylabel('Directors')
plt.show()

# plot correlation matrix for 'tmdb_5000_movies.csv' dataset
corr_matrix = movies_df.corr()
plt.matshow(corr_matrix)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.show()