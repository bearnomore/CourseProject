
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
import time
from scipy import spatial


# In[109]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
# Load transaction dataset
D = pd.read_csv(path + "DBLP2000.csv").astype('str')
D


# In[110]:


# Load context units
context_units = pd.read_csv(path + "DBLP2000_context_units.csv")
context_units


# ### Calculated the weight (Mutual Intformation between each pair of context unit and title transaction)

# In[132]:


D_size = len(D) # number of rows
C_size = len(context_units) # number of columns
# Initialize the weight matrix 
transaction_w = np.zeros([C_size, D_size])
context_index = context_units['transaction_index'].apply(lambda row: row[1:-1].split(', ')).apply(lambda row: list(map(int, row)))
for i in range(C_size):
    context_ind = context_index.iloc[i]
    transaction_w[i, context_ind] = 1


# In[145]:


transaction_weights = pd.DataFrame(transaction_w, index = context_units["pattern"])
transaction_weights


# In[117]:


transaction_w


# In[13]:


transaction_weights.to_csv(path + "transaction_weights.csv", index = False)


# ### Calculate similarity between the given transaction and the author 

# In[135]:


# Get author list 
authors = pd.read_csv(path + "authorsFP2000_with_index.csv")['author']
authors


# In[136]:


# Get author weights
context_weights = pd.read_csv(path+ "Context_units_weights.csv")
author_weights = context_weights.iloc[:, 1:15]
author_weights


# In[137]:


# Define the cosine similarity between two vectors 
def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)


# In[147]:


def sim_tran_author(transaction_weights, author_weights):
    
    num_tran = transaction_weights.shape[1]
    num_author = author_weights.shape[1]
    
    sim_scores = pd.DataFrame(columns = author_weights.columns, index = transaction_weights.columns)
    for t in range(num_tran):
        for a  in range(num_author):
            #features_ind = (~transaction_weights.iloc[:,t].isna()) & (~author_weights.iloc[:,a].isna())
            tran =  transaction_weights.iloc[:, t]
            author = author_weights.iloc[:, a]
         
            #sim_scores.iloc[t,a] = cos_sim(tran, author)
            sim_scores.iloc[t,a]  = 1 - spatial.distance.cosine(tran, author)
    return sim_scores


# In[148]:


sim_scores = sim_tran_author(transaction_weights, author_weights)


# In[153]:


sim_scores.to_csv(path+"similarity_scores_of_transaction_to_author.csv", index = False)


# ### Show representative titles for a given author

# In[154]:


author = authors[0]
r = 10
ind = sim_scores.sort_values(by = author, ascending = False).index[0:r]
rep_titles = pd.DataFrame(columns  = [author])
rep_titles[author] = D['title'].iloc[ind]
rep_titles


# In[155]:


rep_titles.to_csv(path + 'rep_titles_example1.csv', index = False)


# In[156]:


author = authors[5]
r = 10
ind = sim_scores.sort_values(by = author, ascending = False).index[0:r]
rep_titles = pd.DataFrame(columns  = [author])
rep_titles[author] = D['title'].iloc[ind]
rep_titles


# In[157]:


author = authors[10]
r = 10
ind = sim_scores.sort_values(by = author, ascending = False).index[0:r]
rep_titles = pd.DataFrame(columns  = [author])
rep_titles[author] = D['title'].iloc[ind]
rep_titles

