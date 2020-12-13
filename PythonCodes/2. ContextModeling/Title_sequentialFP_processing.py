
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# From title sequential frequrnt pattern mining by titles_seqPattern_mining.py,
# the seqFPs of titles were output as file "part-00000" that needed further processing. 
path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/output/"
titles_spark = pd.read_table(path + "part-00000.txt", header = None)
print(len(titles_spark))
titles_spark.head()


# In[3]:


# Cleaned the data by separating the pattern and frequency into two columns of dataframe
titles = titles_spark[0].apply(lambda x: x.split(" freq"))
title_patterns = titles.apply(lambda x: x[0][16:-4].split("'], ['"))
titles_freq = titles.apply(lambda x:x[1][1:-1]).astype("int")


# In[4]:


title_patterns


# In[5]:


titles_freq


# In[6]:


data = {"title_pattern": title_patterns, 
        "Freq": titles_freq 
        } 
titles_FP_spark = pd.DataFrame(data)


# In[7]:


titles_FP_spark['title_pattern'] = titles_FP_spark.apply(lambda x: ' '.join(x['title_pattern']), axis = 1)
titles_FP_spark.head()


# In[ ]:


output_path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'
titles_FP_spark.to_csv(output_path+"titlesFP2000.csv", index = False)

