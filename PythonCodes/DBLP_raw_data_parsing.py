
# coding: utf-8

# In[1]:


import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/DBLP/" # File path
tree = ET.parse(path+"dblp50000.xml")
root = tree.getroot()


# In[3]:


# Check what tags the root of xml tree has
children = root.getchildren()
tag_list = []
for c in children:
    tag_list.append(c.tag)
tags = set(tag_list)
# Each tag is a type of publication (e.g. article, book, etc) 
# and has similar structure (e.g. having author and title of that publication).
print(tags)


# In[4]:


# Extract all titles
title_list = []
for child in root:
    # By checking the data, some publication had no author, these publications would be skipped. 
    if len(child.findall("author"))==0: 
        continue
    for title in child.findall("title"):
        title_list.append(title.text)


# In[5]:


# Extract all years
year_list = []
for child in root:
    if len(child.findall("author"))==0:
        continue
    for year in child.findall("year"):
        year_list.append(int(year.text))


# In[7]:


# Extract all authors. Some publication had more than one author, 
# all co-authors for each publication would be extracted as a string separated by a comma.
author_list = []
for child in root:
    authors = []
    if len(child.findall("author"))==0:
        continue
    for author in child.findall("author"):
        authors.append(author.text)
        authors_str = ", ".join(authors)
    author_list.append(authors_str)


# In[8]:


# Length of title list, author list and year list should be the same 
# and less than the length of raw data as some entries with no author were not selected.

print(len(title_list), len(author_list), len(year_list))


# In[9]:


# Combine all lists into dblp dataframe 
dblp = pd.DataFrame(columns = ["author", "title", "year"])
dblp["author"] = author_list
dblp["title"] = title_list
dblp["year"] = year_list

dblp.head()


# In[11]:


# For simplicity, we only picked transactions of year 2000
dblp2000 = dblp.loc[dblp['year']==2000, ['author', 'title']]
dblp2000.head()


# In[12]:


output_path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'
dblp2000.to_csv(output_path+"DBLP2000.csv", index = False)

