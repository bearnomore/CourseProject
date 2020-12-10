
# coding: utf-8

# In[13]:


import pandas as pd
path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"


# In[21]:


# Load annotation, representative title and synonyms of the example author
annotation = pd.read_csv(path + "author_annotation_example1.csv").iloc[:, 0].tolist()
rep_title = pd.read_csv(path + "rep_titles_example1.csv").iloc[:, 0].tolist()
syn = pd.read_csv(path + "synonyms_to_author_example1.csv").iloc[:, 0].tolist()


# In[22]:


author_annotation


# In[23]:


annotation


# In[31]:


author_annotation = dict()
author_annotation["author_name"] = 'Ralf Steinmetz'
author_annotation["annotation"] = ', '.join(annotation)
author_annotation["representative_titles"] = ', '.join(rep_title)
author_annotation["synonyms"] = ', '.join(syn)


# In[47]:


author_context_annotation = pd.DataFrame(author_annotation, index = [0])
author_context_annotation.transpose()


# In[46]:


author_context_annotation.transpose().to_csv(path+"context_annotation_example1.csv")

