
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[4]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
Weights = pd.read_csv(path + "Context_units_weights.csv")
Weights


# In[10]:


def annotation(pattern, weights, K):
    '''
    Given an pattern from the context units, the context units weight matrix and K,
    Output the list of top patterns that have the highest K weights (MI scores) with this author
    '''
    annotation = dict()
    annotation[pattern] = weights.sort_values(by = pattern, ascending = False).iloc[0:K]["pattern"]
    return annotation


# In[18]:


# Example, given a frequent author, output the the top 10 context units as its annotation
author_list = Weights.columns[1:]
author = author_list[0]
rank = 10
author1_annotation = pd.DataFrame(annotation(author, Weights, rank))
author1_annotation


# In[15]:


# Example, given a frequent author, output the the top 10 context units as its annotation
title_list = Weights.columns[15:]
title = title_list[10]
rank = 10
pd.DataFrame(annotation(title, Weights, rank))


# In[19]:


output_path = path
author1_annotation.to_csv(output_path + "author_annotation_example1.csv", index = False)

