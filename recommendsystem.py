import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie_id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
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


df = pd.merge(items, ratings, on='movie_id')


print(df.shape)
print(df.head())

print(df['movie title'].unique())

print(df.describe()
)


ratingsavrg = pd.DataFrame(df.groupby('movie title')['rating'].mean())
print(ratingsavrg.head())


ratingsavrg['number_of_ratings'] = df.groupby('movie title')['rating'].count()
print(ratingsavrg.head())

ratingsavrg['rating'].hist(bins=50)
#plt.show()

ratingsavrg['number_of_ratings'].hist(bins=50)

#plt.show()

sns.jointplot(x='rating', y='number_of_ratings', data=ratingsavrg)

#plt.show()


movie_matrix = df.pivot_table(index='user_id', columns='movie title', values='rating')
print(movie_matrix.head())

exit()
print(ratingsavrg.sort_values('number_of_ratings', ascending=False).head(10))

AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']

print(AFO_user_rating.head())
print(contact_user_rating.head())

similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)

print(similar_to_air_force_one.head())

similar_to_contact = movie_matrix.corrwith(contact_user_rating)
print(similar_to_contact.head())

# to drop NaN values
corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
print(corr_contact.head())
corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)
print(corr_AFO.head())



corr_AFO = corr_AFO.join(ratingsavrg['number_of_ratings'])
corr_contact = corr_contact.join(ratingsavrg['number_of_ratings'])
corr_AFO .head()
corr_contact.head()

print(corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10))

print(corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))