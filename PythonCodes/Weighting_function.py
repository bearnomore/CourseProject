
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[2]:


path = "E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/"
context_units = pd.read_csv(path+"DBPL2000_context_units.csv")


# In[3]:


context_units


# In[5]:


D = pd.read_csv(path + "DBLP2000.csv") # Read the original transaction dataset
D


# In[93]:


# Suppose we use the context units themselves as FPs to 
# find out their individual weight vectors in the space made of themselves by 
# building weight matrix of pairwised context units. 
# Each weight between the pair of context unit patterns is the mutual information calculated by the formula per paper
# using the probabilities of four cases: 
# p11: prob of unit1 and unit2 both present in the transaction dataset D
# p10: prob of unit1 present and unit2 absent in the transaction dataset D
# p01: prob of unit1 absent and unit2 present in the transaction dataset D
# p00: of prob of neither unit1 nor unit2 prepsent in the transaction dataset D


# In[8]:


D_size = len(D)
C_size = len(context_units)

# Initialize the matrix 
W = np.zeros([C_size, C_size])

for i in range(C_size):
    for j in range(C_size):
        ind1 = context_units['transaction_index'].iloc[i]
        ind2 = context_units['transaction_index'].iloc[j]
        intersection = set(ind1).intersection(ind2)
        #Calculate the probabilities with laplace smoothing
        p11 = (len(intersection)+0.25)/(D_size + 1)
        p01 = (len(ind2)- len(intersection) + 0.25)/(D_size + 1)
        p10 = (len(ind1)- len(intersection) + 0.25)/(D_size + 1)
        p00 = 1 - p11 - p01 - p10
        
        su1 = len(ind1)/D_size #support of u1
        su2 = len(ind2)/D_size #support of u2
        nu1 = 1-su1
        nu2 = 1-su2
        
        MI = p11*np.log10(p11/su1/su2) +              p01*np.log10(p01/nu1/su2) +              p10*np.log10(p10/su1/nu2) +              p00*np.log10(p00/nu1/nu2)
        W[i,j] = MI


# In[9]:


W


# In[11]:


Weights = pd.DataFrame(W, columns = context_units["pattern"], index = context_units["pattern"])
Weights


# In[12]:


output_path = path
Weights.to_csv(output_path + "Context_units_weights.csv", index = False)

