#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[41]:


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


# In[42]:


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
#     if grp_ids:
#         D = Xtr[grp_ids].toarray()
#     else:
    D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


# In[43]:


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


# In[44]:


def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# In[45]:


keywords= pd.read_csv('./keywords.csv')
# print(len(keywords))
keywords.drop_duplicates(keep=False,inplace=True) 
print(len(keywords))
credits= pd.read_csv('./credits.csv')
# print(len(credits))
credits.drop_duplicates(keep=False,inplace=True) 
print(len(credits))
moviestokeep= pd.read_csv('./links_small.csv')


movies= pd.read_csv('./movies_metadata.csv')
movies.drop_duplicates(keep=False,inplace=True) 
print(len(movies))


# In[46]:


idstokeep= list(moviestokeep.loc[:, 'tmdbId'])
idstokeep2= list(keywords.loc[:, 'id'])
idstokeep3= list(credits.loc[:, 'id'])

keywords2= keywords[keywords['id'].isin(idstokeep)]

idstokeep4= list(keywords2.loc[:, 'id'])

credits2= credits[credits['id'].isin(idstokeep4)]
idstokeep5= list(keywords2.loc[:, 'id'].apply(str))
movies2=movies[movies['id'].isin(idstokeep5)]
# movies2['id'] = movies2['id'].apply(str)
print(len(keywords2))
print(len(credits2))
print(len(movies2))


# In[47]:


pal=keywords2['id']
pal=np.array(pal)

sal=credits2['id']
sal=np.array(sal)

tal=movies2['id']
tal=np.array(tal)

# print(len(np.intersect1d(pal,tal)))


# In[48]:


clist=credits2['cast']
klist=keywords2['keywords']
mlist=movies2['original_title']
# print(mlist)


# In[49]:


clist=np.array(clist)
castlist=[]
for j in range(0,len(clist)):
    tempstr=clist[j]
    tg=[i for i in range(len(tempstr)) if tempstr.startswith('name\':', i)]
    th=[i for i in range(len(tempstr)) if tempstr.startswith('order\':', i)]
    tempsr=""
    for k in range(0,len(tg)):
        temps=tempstr[tg[k]+8:th[k]-4]
        tempss=temps.replace(" ", "")
        tempsr=tempsr+tempss+" "
    tempsr=tempsr.strip()
    castlist.append(tempsr)


# In[50]:


print(castlist[0])


# In[51]:


klist=np.array(klist)
keylist=[]
for j in range(0,len(klist)):
    tempstr=klist[j]
    tg=[i for i in range(len(tempstr)) if tempstr.startswith('name\':', i)]
    th=[i for i in range(len(tempstr)) if tempstr.startswith('}', i)]
    tempsr=""
    for k in range(0,len(tg)):
        temps=tempstr[tg[k]+8:th[k]-1]
        tempss=temps.replace(" ", "")
        tempsr=tempsr+tempss+" "
    tempsr=tempsr.strip()
    keylist.append(tempsr)


# In[52]:


print(keylist[0])


# In[53]:


mlist=np.array(mlist)
movlist=[]
for j in range(0,len(mlist)):
    tempstr=mlist[j]
    movlist.append(tempstr)


# In[54]:


print(movlist[0])


# In[55]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(castlist)
print(X.shape)


# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
Y = vectorizer.fit_transform(keylist)
print(Y.shape)


# In[60]:


CCS=[]
for i in range(0,Y.shape[0]):
    CCS.append(cosine_similarity(Y[i:i+1],Y)[0])
CCS=np.array(CCS)


# In[61]:


KCS=[]
for i in range(0,X.shape[0]):
    KCS.append(cosine_similarity(X[i:i+1],X)[0])
KCS=np.array(KCS)
print(KCS)


# In[62]:


CS=0.7*KCS+0.3*CCS
with open('./CS.pkl', 'wb+') as f:
    pickle.dump(CS, f)


# In[3]:


movino=0
intresarr=CS[movino]
mvnameintr=movlist[movino]
intresarr=np.array(intresarr)
indxar=np.argsort(intresarr)[-10:]
# print(indxar)
k=[]
x=[]
y=[]
for i in indxar:
    k.append(movlist[i])
    x.append(KCS[movino][i])
    y.append(CCS[movino][i])
# width of the bars
barWidth = 1
# Choose the height of the blue bars
bars1 = [10, 9, 2]

# Choose the height of the cyan bars
bars2 = [10.8, 9.5, 4.5]

# Choose the height of the error bars (bars1)
yer1 = [0.5, 0.4, 0.5]

# Choose the height of the error bars (bars2)
yer2 = [1, 0.7, 1]

# The x position of bars
r1 = 2.5*np.arange(len(x))
r2 = [i + barWidth for i in r1]
#     print(r1)
#     print(r2)

# Create blue barsnp.
plt.barh(r2, y,  color = 'red', edgecolor = 'black', capsize=10, label='Cast Similarity')
plt.barh(r1, x, color = 'blue', edgecolor = 'black', capsize=10, label='Keyword Similarity')

# Create cyan bars


# general layout
plt.yticks([r + barWidth for r in r1], k)
plt.xlabel('Similarity')
plt.legend()
plt.title('Top Similar Movies like '+mvnameintr)
# Show graphic
plt.show()


# In[ ]:




