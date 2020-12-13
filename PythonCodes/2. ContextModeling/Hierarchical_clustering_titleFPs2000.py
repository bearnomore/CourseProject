
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.cluster as cluster
import scipy.spatial.distance as ssd
import time
from matplotlib import pyplot as plt


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
titleFPs = pd.read_csv(path + "titlesFP2000_with_index.csv")


# In[3]:


titleFPs.sort_values('Freq', ascending = False)


# In[4]:


# Define the Jaccard_Dist per paper
def Jaccard_Dist(p1_indice, p2_indice):
    '''
    Given transaction indice list of two patterns, p1_indice and p2_indice,
    Output the Jaccard distance defined as 1- intersection(p1_indice, p2_indice)/union(p1_indice, p2_indice)
    '''
    return 1-len(set(p1_indice).intersection(p2_indice))/len(set(p1_indice).union(p2_indice))


# In[5]:


# Example of Jaccard distance betwen two title FPs
p1 = titleFPs.iloc[0]['transaction_index']
p2 = titleFPs.iloc[1]['transaction_index']
Jaccard_Dist(p1, p2)


# In[6]:


# Jaccard Distance matrix between pariwise title FPs
np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})
num_ind =  len(titleFPs)
c = 0
# J_dist is a symetric num_ind x num_ind matrix 
# and the diagnal of the matrix is the distance between the title pattern and itself,
# therefore the diagnal of the matrix should be all 0. 
J_dist = np.zeros([num_ind,num_ind]) 
start =  time.time()
while c < num_ind:
    
    p = titleFPs['transaction_index'].iloc[c]
    # Fill the upper and lower triangle of the matrix 
    J_dist[c+1:, c] = titleFPs['transaction_index'].iloc[c+1:].apply(lambda x: Jaccard_Dist(p, x))
    J_dist[c, c+1:] = titleFPs['transaction_index'].iloc[c+1:].apply(lambda x: Jaccard_Dist(p, x))
    c +=1
    
end = time.time()
print((end - start)/60)


# In[7]:


J_dist.mean()


# In[10]:


# Get 1D condensed array of J_dist
J_distArray = ssd.squareform(J_dist)
# Perform clustering using complete linkage
Z = cluster.hierarchy.complete(J_distArray)


# In[11]:


# Check the first 20 iteratives of the clustering 
Z[:20]


# In[12]:


# Display full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
cluster.hierarchy.dendrogram(
    Z,
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    color_threshold = 0.8*max(Z[:,2]),
)
plt.show()


# In[13]:


# Display only the top branches 
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
cluster.hierarchy.dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    color_threshold = 0.8*max(Z[:,2]),
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


# In[14]:


last = Z[-200:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2 
print(k)


# In[15]:


# Get clusters
max_d = 0.01
clusters = cluster.hierarchy.fcluster(Z, max_d, criterion='distance')
print("Number of Clusters: ", len(set(clusters)))

# Add cluster labels to the title patterns dataset titles_FPs
titleFPs['cluster_index'] = clusters
titleFPs


# In[16]:


# Group the titles_FPs by cluster index and pick the 'centroid' pattern with the maximum support
titleFPs_centroid = titleFPs.groupby("cluster_index")['title_pattern'].max('Freq')
titleFPs_centroid


# In[17]:


titlesFP2000_final = titleFPs.loc[titleFPs['title_pattern'].isin(titleFPs_centroid)].sort_values(by= 'Freq', ascending = False)


# In[18]:


titlesFP2000_final.head()


# In[ ]:


output_path = path
titlesFP2000_final.to_csv(output_path + "titlesFP2000_final.csv", index = False)


# In[ ]:


# dist_clusters = Z[:,2]
# idxs = np.arange(1, len(dist_clusters) + 1)
# plt.plot(idxs, dist_clusters, 'b')
# acceleration = np.diff(dist_clusters, 2)  # 2nd derivative of the distances
# plt.plot(idxs[:-2] + 1, acceleration, 'r')
# plt.show()
# k = acceleration.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
# print("clusters:", k)

