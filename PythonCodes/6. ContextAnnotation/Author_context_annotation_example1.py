
# coding: utf-8

# In[1]:


import pandas as pd
path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"


# In[2]:


# Load annotation, representative title and synonyms of the example author
definition = pd.read_csv(path + "author_annotation_example1.csv")
rep_title = pd.read_csv(path + 'rep_titles_author_example1.csv')
syn_author = pd.read_csv(path + "coauthor_to_author_example1.csv")
syn_title = pd.read_csv(path + "syn_titles_to_author_example1.csv")


# In[3]:


definition


# In[4]:


rep_title


# In[5]:


syn_author


# In[6]:


syn_title


# In[7]:


author = syn_title.columns.values[0]
Author1_annotation = pd.DataFrame(columns = ['Author','Definition', 'Representative Titles', 'Synonym Authors', 'Synonym Titles'])
Author1_annotation['Author'] = [author] * 5
Author1_annotation['Definition'] = definition
Author1_annotation['Representative Titles'] = rep_title
Author1_annotation['Synonym Authors'] = syn_author
Author1_annotation['Synonym Titles'] = syn_title
Author1_annotation


# In[ ]:


Author1_annotation.to_csv(path+"context_annotation_author1.csv", index =  False)

