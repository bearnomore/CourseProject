
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
Weights = pd.read_csv(path + "Context_units_weights.csv")
Weights


# In[3]:


def annotation(pattern, weights, K):
    '''
    Given an pattern from the context units, the context units weight matrix and K,
    Output the list of top patterns that have the highest K weights (MI scores) with this author
    '''
    annotation = dict()
    annotation[pattern] = weights.sort_values(by = pattern, ascending = False).iloc[0:K]["pattern"]
    return annotation


# In[4]:


# Example1, given a frequent author, output the the top 5 context units as its annotation
author_list = Weights.columns[1:]
author = author_list[0]
rank = 5
author1_annotation = pd.DataFrame(annotation(author, Weights, rank))
author1_annotation


# In[5]:


# Example2, given a frequent title, output the the top 5 context units as its annotation
title_list = Weights.columns[16:]
title = title_list[10]
rank = 5
title1_annotation = pd.DataFrame(annotation(title, Weights, rank))
title1_annotation


# In[ ]:


output_path = path
author1_annotation.to_csv(output_path + "author_annotation_example1.csv", index = False)
title1_annotation.to_csv(output_path + "title_annotation_example1.csv", index = False)

