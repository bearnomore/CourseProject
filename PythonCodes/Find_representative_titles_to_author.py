
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
# Load transaction dataset
D = pd.read_csv(path + "DBLP2000.csv").astype('str')
D


# In[40]:


# Load context units
context_units = pd.read_csv(path + "DBLP2000_context_units.csv")
context_units


# ### Calculated the weight (Mutual Intformation between each pair of context unit and title transaction)

# In[55]:


start = time.time()

# First build probability matrices of all four cases per definition in paper
# p11: context = 1 and transaction = 1
# p01: context = 0, and transaction = 1
# p10: context = 1, and transaction = 0
# p00: context = 0, and transaction = 0

D_size = len(D) # number of rows
C_size = len(context_units) # number of columns

# Initialize the weight matrix 
transaction_w = np.zeros([C_size, D_size])
intersection = 0
context_index = context_units['transaction_index'].apply(lambda row: row[1:-1].split(', ')).apply(lambda row: list(map(int, row)))

for i in range(C_size):
    context_ind = context_index.iloc[i]
    for j in range(D_size):
        if j in context_ind:
            intersection = 1
        p11 = (intersection +0.25)/(D_size + 1)
        p01 = (1-intersection+0.25)/(D_size + 1)
        p10 = (len(context_ind) - intersection + 0.25)/(D_size + 1)
        p00 = 1-p11-p01-p10
        
        sc1 = len(context_ind)/D_size # support of context unit
        sc0 = 1 - sc1
        
        st1 = 1/D_size # support of transaction
        st0 = 1-st1
            
        MI = p11*np.log10(p11/sc1/st1) +              p01*np.log10(p01/sc0/st1) +              p10*np.log10(p10/sc1/st0) +              p00*np.log10(p00/sc0/st0)
        
        transaction_w[i,j] = MI
        
end = time.time()
print((end-start)/60)


# In[56]:


transaction_w


# In[57]:


transaction_weights = pd.DataFrame(transaction_w, index = context_units["pattern"])
transaction_weights


# In[13]:


transaction_weights.to_csv(path + "transaction_weights.csv", index = False)


# ### Calculate similarity between the given transaction and the author 

# In[58]:


# Get author list 
authors = pd.read_csv(path + "authorsFP2000_with_index.csv")['author']
authors


# In[59]:


# Get author weights
context_weights = pd.read_csv(path+ "Context_units_weights.csv")
author_weights = context_weights.iloc[:, 1:15]
author_weights


# In[60]:


# Define the cosine similarity between two vectors 
def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)


# In[61]:


def sim_tran_author(transaction_weights, author_weights):
    
    num_tran = transaction_weights.shape[1]
    num_author = author_weights.shape[1]
    
    sim_scores = pd.DataFrame(columns = author_weights.columns, index = transaction_weights.columns)
    for t in range(num_tran):
        for a  in range(num_author):
            #features_ind = (~transaction_weights.iloc[:,t].isna()) & (~author_weights.iloc[:,a].isna())
            tran =  transaction_weights.iloc[:, t]
            author = author_weights.iloc[:, a]
            sim_scores.iloc[t,a] = cos_sim(tran, author)
    return sim_scores


# In[62]:


sim_scores = sim_tran_author(transaction_weights, author_weights)


# In[63]:


sim_scores


# In[32]:


sim_scores.to_csv(path+"similarity_scores_of_transaction_to_author.csv", index = False)


# ### Show representative titles for a given author

# In[72]:


author = authors[0]
r = 10
ind = sim_scores.sort_values(by = author, ascending = False).index[0:r]
rep_titles = pd.DataFrame(columns  = [author])
rep_titles[author] = D['title'].iloc[ind]
rep_titles


# In[73]:


rep_titles.to_csv(path + 'rep_titles_example1.csv', index = False)

