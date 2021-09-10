#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
import math
from sklearn.metrics import mean_squared_error 
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


def cosine_similarity(ui, uk):
    norm1= np.linalg.norm(ui)
    norm2= np.linalg.norm(uk)
    return (np.dot(ui, uk.T)/(norm1*norm2))


# In[11]:


def get_recommendation(user, utility, u, s, vt):#recommendation for ith user
    scoredict= {}
    num= np.sum([e!= 0 for e in utility[user]])
    scorepapa= np.sum(utility[user])/num
    for i in range(len(utility[0])):
        score= 0
        summ= 0
        for j in range(len(utility)):
            if(utility[user][i]== 0 and utility[j][i]!= 0):
                sim= cosine_similarity(u[user], u[j])
                num2= np.sum([e!= 0 for e in utility[j]])
                score2= np.sum(utility[j])/num2
                score+= sim*abs((utility[j][i] - score2))
                summ+= sim
        
        if(summ!= 0):
            scoredict[i]= score/summ + scorepapa
#         else:
#             scoredict[i]= scorepapa
    
    sorteddict= sorted(scoredict.items(), key= lambda kv: kv[1], reverse= True)
    sorteddict= sorteddict[:5]#top 5 values
    return sorteddict
    


# In[12]:


def get_movies(sorteddict, dictids):
    movieids= []
    for i in range(len(sorteddict)):
        movieids.append([dictids[sorteddict[i][0]], sorteddict[i][1]])
    
    return movieids
    


# In[5]:


ratings= pd.read_csv('./ratings_small.csv')
print (ratings.columns)
moviedata= pd.read_csv('./movies_metadata.csv')
moviedata.drop(['budget', 'homepage', 'original_language', 'overview', 'runtime', 'video', 'spoken_languages', 
                'production_countries', 'release_date', 'poster_path', 'production_companies', 'revenue', 
                'popularity', 'status', 'tagline', 'vote_count'], axis= 1)
print (moviedata.columns)
print (len(moviedata))
moviestokeep= pd.read_csv('./links_small.csv')
print (moviestokeep.columns)


# In[16]:


# moviedata.drop_duplicates(keep=False,inplace=True)
# print (len(moviedata))
# print (moviestokeep.columns)
# idstokeep= list(moviestokeep.loc[:, 'tmdbId'])
# print (len(idstokeep))
# moviedata2= moviedata[moviedata['id'].isin(idstokeep)]
# print (len(moviedata2))

def get_moviename(movieids, moviedata, user):
    print  ("Movies to be recommended to User " + str(user+1) + " are as follows:- ")
    print ()
    for i in range(len(movieids)):
        print (moviedata.loc[movieids[i][0], 'original_title'], movieids[i][1])
    return
    
    


# In[7]:


print (len(list(ratings['userId'])))
a= np.unique(np.array(ratings['movieId']))
print (len(a))
dictids= {}
for i in range(len(a)): 
    dictids[i]= a[i]


# In[8]:


#creating the utility matrix
# utility= np.zeros([671, 9066])
# for i in range(len(utility)):
#     for j in range(len(utility[0])):
#         temp= ratings.loc[ratings['userId']== i+1]
#         xx= temp.loc[temp['movieId']== int(dict[j]), 'rating']
#         #print (xx)
#         if(len(xx)!= 0):
#             utility[i][j]= xx.iloc[0]


# print (np.sum(utility[0]))
# print (np.unique(utility[0]))

# utility.dump('./utility.dat')


# In[9]:


utility= np.load('./utility.dat')
u, s, vt= svds(utility, k= 25) #reduce to 50 latent factors
print (u.shape, s.shape, vt.shape)
#print (max(utility[]))


# In[17]:


ids= get_recommendation(2, utility, u, s, vt)
movieids= get_movies(ids, dictids)
get_moviename(movieids, moviedata, 0)


# In[ ]:


#claculate the rmse for our recommender
s= np.diag(s)
print (math.sqrt(mean_squared_error(utility, np.dot(np.dot(u, s), vt))))


# In[ ]:


errors= []
factors= np.arange(1, 26)
for i in range(len(factors)):
    u, s, vt= svds(utility, k= factors[i])
    print (u.shape, s.shape, vt.shape)
    s= np.diag(s)
    errors.append(math.sqrt(mean_squared_error(utility, np.dot(np.dot(u, s), vt))))

plt.scatter(factors, errors, marker= 'x')
plt.xlabel('Number of Latent Factors')
plt.ylabel('RMSE for the Model')
plt.title('Latent Factors v/s RMSE')
plt.show()


# In[ ]:




