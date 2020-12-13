
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
# Load context units wieghts
context_weights = pd.read_csv(path + "Context_units_weights.csv")
context_weights


# In[3]:


# Extract the author weights
author_weights = context_weights.iloc[:, 1:15]
author_weights


# In[4]:


# Extract the title weights
title_weights = context_weights.iloc[:, 15:]
title_weights


# In[5]:


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


def sim_candidate_pattern(candidate_weights, pattern_weights):
    
    num_can = candidate_weights.shape[1]
    num_pattern = pattern_weights.shape[1]
    
    sim_scores = pd.DataFrame(columns = pattern_weights.columns, index = candidate_weights.columns)
    for c in range(num_can):
        for p  in range(num_pattern):
            can =  candidate_weights.iloc[:, c]
            pattern = pattern_weights.iloc[:, p]
            sim_scores.iloc[c,p] = cos_sim(can, pattern)
    return sim_scores


# ### Find author patterns which are most semantically similar to the given author

# In[10]:


# Let the whole author FP be the candidates
sim_scores_authors = sim_candidate_pattern(author_weights, author_weights)
sim_scores_authors


# In[ ]:


sim_scores_authors.to_csv(path+"similarity_scores_of_author_candidates_to_author.csv", index = False)


# In[11]:


# Find synonym (coauthor) of a given author
r = 6
author = author_weights.columns[0] # pick the first author
co_authors = pd.DataFrame(columns = [author])
co_authors[author] = sim_scores_authors.sort_values(by = author, ascending = False).index[1:r]
co_authors


# In[ ]:


co_authors.to_csv(path + "coauthor_to_author_example1.csv", index = False)


# ### Find title patterns which are most semantically similar to the given author

# In[12]:


# Let the whole author FP be the candidates
sim_scores_titles = sim_candidate_pattern(title_weights, author_weights)
sim_scores_titles


# In[ ]:


sim_scores_titles.to_csv(path+"similarity_scores_of_title_candidates_to_author.csv", index = False)


# In[13]:


# Find synonym (titles) of a given author
r = 5
author = author_weights.columns[0] # pick the first author
titles = pd.DataFrame(columns = [author])
titles[author] = sim_scores_titles.sort_values(by = author, ascending = False).index[0:r]
titles


# In[ ]:


titles.to_csv(path + "syn_titles_to_author_example1.csv", index = False)


# ### Find title patterns which are most semantically similar to the given title

# In[14]:


sim_scores_TT = sim_candidate_pattern(title_weights, title_weights)
sim_scores_TT


# In[ ]:


sim_scores_TT.to_csv(path+"similarity_scores_of_title_candidates_to_title.csv", index = False)


# In[15]:


# Find synonym (titles) of a given author
r = 6
title = title_weights.columns[11] # pick the same title in definition
titles2 = pd.DataFrame(columns = [title])
titles2[title] = sim_scores_TT.sort_values(by = title, ascending = False).index[1:r]
titles2


# In[ ]:


titles2.to_csv(path + "syn_titles_to_title_example1.csv", index = False)


# ### Find author patterns which are most semantically similar to the given title

# In[16]:


# Find synonym (titles) of a given author
r = 5
title = title_weights.columns[11] # pick the first author
titles3 = pd.DataFrame(columns = [title])
titles3[title] = sim_scores_titles.transpose().sort_values(by = title, ascending = False).index[0:r]
titles3


# In[ ]:


titles3.to_csv(path + "syn_authors_to_title_example1.csv", index = False)

