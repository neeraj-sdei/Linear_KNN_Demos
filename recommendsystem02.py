import pandas as pd
import numpy as np

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie_id', 'movie_title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')


print(users.shape)
print(items.shape)
print(users.head())
print(items.head())
print(ratings.head())
print(ratings.shape)

ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

print(ratings_train.head())
ratings_train = pd.merge(ratings_train,items[['movie_id','movie_title']],on='movie_id', how='left')
ratings_test = pd.merge(ratings_test,items[['movie_id','movie_title']],on='movie_id', how='left')
print(ratings_train.head())


print(ratings_train.shape, ratings_test.shape)

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

print(user_prediction)

import turicreate
train_data = turicreate.SFrame(ratings_train)
test_data = turicreate.SFrame(ratings_test)


#First we’ll build a model which will recommend movies based on the most popular choices,
# i.e., a model where all the users receive the same recommendation(s)
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
#It’s prediction time! We will recommend the top 5 items for the first 5 users in our dataset.
popularity_recomm.print_rows(num_rows=25)


#------------------------------------------------------------#-----------------------

#After building a popularity model, we will now build a collaborative filtering model
#Training the model
item_sim_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_title', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=[1,2,3,4,5],k=5)
#df = pd.merge(items, item_sim_recomm.to_dataframe(), on='movie_id')
#print(df['user_id'].head(25))
#item_sim_recomm.print_rows(num_rows=25)

item_sim_model.save("recommend_movies")
#Here we can see that the recommendations (movie_id) are different for each user. So personalization exists,
# i.e. for different users we have a different set of recommendations