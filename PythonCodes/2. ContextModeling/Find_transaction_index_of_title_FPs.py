
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
titlesFP = pd.read_csv(path + "titlesFP2000.csv") # closed sequential patterns found by PySpark PrefixSpan algorithm
dblp_titles = pd.read_table(path + "DBLP2000_preprocessed_titles.txt", header = None) # preprocess titles from DBLP dataset


# In[3]:


titlesFP['title_pattern'] = titlesFP['title_pattern'].astype('str')
dblp_titles.columns = ['title']
dblp_titles['title'] = dblp_titles['title'].astype('str')


# In[4]:


titlesFP.sort_values('Freq', ascending = False)


# In[5]:


dblp_titles


# In[6]:


def find_pattern(title_pattern, title):
    output = False
    if title.find(title_pattern) != -1:
        output = True
    if title.find(title_pattern) == -1: 
        tp = title_pattern.split(' ')
        if len(tp) > 1 and all(title.find(p) != -1 for p in tp):
            output = True
    return output


# In[7]:


# Find index of a title pattern in the dblp titles
def pattern_index(p, dblp_titles):
    '''
    Given a closed sequential pattern p of titles and the preprocess dblp_titles dataset,
    Output the indice of the transactions that include the pattern, p.
    '''
    
    return dblp_titles.loc[dblp_titles['title'].apply(lambda t: find_pattern(p, t))].index.tolist()


# In[8]:


# Find index of all title patterns in the title_spark dataset
start =  time.time()
titlesFP["transaction_index"] = titlesFP['title_pattern'].apply(lambda x: pattern_index(x, dblp_titles))
end = time.time()
print((end-start)/60)


# In[9]:


titlesFP


# In[10]:


# Check if all title patterns have transaction index found
titlesFP.loc[titlesFP['transaction_index'].apply(lambda x: len(x)==0)]


# In[11]:


# Remove the above title pattern
broken = titlesFP.loc[titlesFP['transaction_index'].apply(lambda x: len(x)==0)].index.values
titlesFP = titlesFP.drop(broken)


# In[12]:


titlesFP


# In[ ]:


output_path = path
titlesFP.to_csv(output_path + "titlesFP2000_with_index.csv", index = False)

