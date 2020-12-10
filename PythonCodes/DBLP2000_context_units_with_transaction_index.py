
# coding: utf-8

# In[12]:


# Combine author and title FPs dataset to form the final context units dataset with transaction index
import pandas as pd

path  = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'

authorFPs = pd.read_csv(path + 'authorsFP2000_with_index.csv')
titleFPs = pd.read_csv(path + 'titlesFP2000_final.csv')


# In[13]:


authorFPs


# In[14]:


titleFPs


# In[15]:


titleFPs_ = titleFPs.iloc[:, [0, 2]]
titleFPs_.columns = ["pattern", "transaction_index"]
titleFPs_["pattern_type"] = "title"

authorFPs_ = authorFPs
authorFPs_.columns = ["pattern", "transaction_index"]
authorFPs_["pattern_type"] = "author"

context_units = pd.concat([authorFPs, titleFPs_])


# In[16]:


output_path = path 
context_units.to_csv(output_path + "DBLP2000_context_units.csv", index = False)

