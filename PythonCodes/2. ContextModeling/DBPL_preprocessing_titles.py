
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd


# In[2]:


# Load parsed DBLP2000 and turn all title into lower case
path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/" # File location
dblp2000 = pd.read_csv(path + "DBLP2000.csv")
dblp2000["title"] = dblp2000["title"].str.lower()
dblp2000.head()


# ### The titles of the DBLP2000 dataset would be stemmed and processed to remove common words. 

# In[3]:


# Tokenization

def identify_tokens(row):
    title = row['title'].lower()
    tokens = nltk.word_tokenize(title)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

dblp2000['words'] = dblp2000.apply(lambda x: identify_tokens(x), axis=1)


# In[4]:


# Stop words removal, there are lots of german publications in the DBLP dataset, 
# thus the stopwords list includes both English and German.
stops = set(stopwords.words(["english", "german"]))                  

def remove_stops(row):
    my_list = row['words']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

dblp2000['meaningful'] = dblp2000.apply(lambda x:remove_stops(x), axis=1)


# In[5]:


# Stemming
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['meaningful']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

dblp2000['meaningful_stemmed_words'] = dblp2000.apply(lambda x: stem_list(x), axis=1)


# In[6]:


dblp2000.head()


# In[7]:


# The processed titles in "meaningful_stemmed_words" column 
# would be saved as the txt file for the next sequential pattern discovery using prefixspan algorithm in PySpark. 
title_seq = dblp2000['meaningful_stemmed_words'].apply(lambda x: ' '.join(x))
title_seq


# In[ ]:


output_path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'
title_seq.to_csv(output_path+"DBLP2000_preprocessed_titles.txt", header = False, index = False)

