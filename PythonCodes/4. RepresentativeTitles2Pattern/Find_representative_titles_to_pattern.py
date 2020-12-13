
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
# Load transaction dataset
D = pd.read_csv(path + "DBLP2000.csv").astype('str')
D


# In[3]:


# Load context units
context_units = pd.read_csv(path + "DBLP2000_context_units.csv")
context_units


# ### Build weight matrix of transactions in the context unit space as bit vectors

# In[4]:


D_size = len(D) # number of rows
C_size = len(context_units) # number of columns
# Initialize the weight matrix 
#transaction_w = np.zeros([C_size, D_size])
transaction_w = -1 * np.ones([C_size, D_size])
context_index = context_units['transaction_index'].apply(lambda row: row[1:-1].split(', ')).apply(lambda row: list(map(int, row)))
for i in range(C_size):
    context_ind = context_index.iloc[i]
    transaction_w[i, context_ind] = 1


# In[5]:


transaction_weights = pd.DataFrame(transaction_w, index = context_units["pattern"])
transaction_weights


# In[ ]:


transaction_weights.to_csv(path + "transaction_weights.csv", index = False)


# ### Calculate similarity between the given transaction and the author 

# In[6]:


# Get author list 
authors = context_units.iloc[0:14]["pattern"]
authors


# In[7]:


# Get author weights
context_weights = pd.read_csv(path+ "Context_units_weights.csv")
author_weights = context_weights.iloc[:, 1:15]
author_weights


# In[8]:


# Define the cosine similarity between two vectors 
def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)


# In[9]:


def sim_tran_pattern(transaction_weights, pattern_weights):
    
    num_tran = transaction_weights.shape[1]
    num_p = pattern_weights.shape[1]
    
    sim_scores = pd.DataFrame(columns = pattern_weights.columns, index = transaction_weights.columns)
    for t in range(num_tran):
        for p  in range(num_p):
            #features_ind = (~transaction_weights.iloc[:,t].isna()) & (~author_weights.iloc[:,a].isna())
            tran =  transaction_weights.iloc[:, t]
            pattern = pattern_weights.iloc[:, p]
         
            #sim_scores.iloc[t,a] = cos_sim(tran, author)
            sim_scores.iloc[t,p]  = 1 - spatial.distance.cosine(tran, pattern)
    return sim_scores


# ### Show representative titles for a given author

# In[10]:


sim_scores_author = sim_tran_pattern(transaction_weights, author_weights)


# In[24]:


sim_scores_author.to_csv(path+"similarity_scores_of_transaction_to_author.csv", index = False)


# In[25]:


author = authors[0]
r = 5
sorted_scores_author = sim_scores_author.sort_values(by = author, ascending = False)
sorted_scores_author.head(10)


# In[26]:


ind = sorted_scores_author.index[0:r]
rep_titles = pd.DataFrame(columns  = [author])
rep_titles[author] = D['title'].iloc[ind]
rep_titles


# In[27]:


rep_titles.to_csv(path + 'rep_titles_author_example1.csv', index = False)


# ### Calculate similarity between the transaction and the title

# In[28]:


# Get title list 
titles = context_units.iloc[14:]["pattern"]
titles


# In[29]:


title_weights = context_weights.iloc[:, 15:]
title_weights


# In[30]:


start = time.time()
sim_scores_title = sim_tran_pattern(transaction_weights, title_weights)
end = time.time()
print((end-start)/60)


# In[31]:


sim_scores_title.to_csv(path+"similarity_scores_of_transaction_to_title.csv", index = False)


# ### Show representative titles for a given title

# In[32]:


title = titles[10+15] # the same one used for definition
r = 5
sorted_scores_title = sim_scores_title.sort_values(by = title, ascending = False)
sorted_scores_title.head(10)


# In[33]:


ind = sorted_scores_title.index[0:r]
rep_titles2 = pd.DataFrame(columns  = [title])
rep_titles2[title] = D['title'].iloc[ind]
rep_titles2


# In[35]:


rep_titles2.to_csv(path + 'rep_titles_title_example1.csv', index = False)

