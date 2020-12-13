
# coding: utf-8

# In[1]:


import pandas as pd
path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"


# In[2]:


# Load annotation, representative title and synonyms of the example author
definition = pd.read_csv(path + "title_annotation_example1.csv")
rep_title = pd.read_csv(path + 'rep_titles_title_example1.csv')
syn_title = pd.read_csv(path + "syn_titles_to_title_example1.csv")
syn_author = pd.read_csv(path + "syn_authors_to_title_example1.csv")


# In[3]:


definition


# In[4]:


rep_title


# In[5]:


syn_author


# In[6]:


syn_title


# In[7]:


title = syn_title.columns.values[0]
Title1_annotation = pd.DataFrame(columns = ['Title','Definition', 'Representative Titles', 'Synonym Titles', 'Synonym Authors'])
Title1_annotation['Title'] = [title] * 5
Title1_annotation['Definition'] = definition
Title1_annotation['Representative Titles'] = rep_title
Title1_annotation['Synonym Authors'] = syn_author
Title1_annotation['Synonym Titles'] = syn_title
Title1_annotation


# In[ ]:


Title1_annotation.to_csv(path+"context_annotation_title1.csv", index = False)

