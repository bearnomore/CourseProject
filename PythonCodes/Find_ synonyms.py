
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import time


# In[7]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
# Load context units wieghts
context_weights = pd.read_csv(path + "Context_units_weights.csv")
context_weights


# In[8]:


# Extract the author weights
author_weights = context_weights.iloc[:, 1:15]
author_weights


# In[9]:


# Define the cosine similarity between two vectors 
def cos_sim(a, b):
    """
    Takes 2 vectors a, b and returns the cosine similarity 
    """
    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)


# In[13]:


def sim_candidate_author(candidate_weights, author_weights):
    
    num_can = candidate_weights.shape[1]
    num_author = author_weights.shape[1]
    
    sim_scores = pd.DataFrame(columns = author_weights.columns, index = candidate_weights.columns)
    for c in range(num_can):
        for a  in range(num_author):
            can =  candidate_weights.iloc[:, c]
            author = author_weights.iloc[:, a]
            sim_scores.iloc[c,a] = cos_sim(can, author)
    return sim_scores


# In[14]:


# Let the whole author FP be the candidates
sim_scores = sim_candidate_author(author_weights, author_weights)
sim_scores


# In[24]:


sim_scores.to_csv(path+"similarity_scores_of_author_candidates_to_author.csv", index = False)


# In[22]:


# Find synonym of a given author
r = 4
author = author_weights.columns[0]
co_authors = pd.DataFrame(columns = [author])
co_authors[author] = sim_scores.sort_values(by = author, ascending = False).index[1:4]
co_authors


# In[23]:


co_authors.to_csv(path + "synonyms_to_author_example1.csv", index = False)

