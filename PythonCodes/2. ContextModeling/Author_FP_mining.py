
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


# In[2]:


# The closed Frequent Patterns of authors would be mined by FP growth algorithm in mlxtend lib + closed pattern definition
path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'
dblp2000 = pd.read_csv(path + "DBLP2000.csv")
dblp2000["author"] = dblp2000.apply(lambda x: x["author"].split(", "), axis = 1) # turn to list of authors for each transaction
dblp2000.head()


# In[3]:


dblp2000['author'].iloc[2]


# In[4]:


# Applied FPgrowth algorithm in MLXend to find frequent patterns given threshold support
dataset = list(dblp2000["author"])
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
thresh = 4/df.shape[0] # used 4 instead of 10 supports as threshold due to smaller dataset
freq_df = fpgrowth(df, min_support=thresh, use_colnames=True)


# In[5]:


freq_df


# #### Find closed frequent itemset using frequent itemset 

# In[6]:


su = freq_df.support.unique() #all unique support count
#Dictionay storing itemset with same support count key
fredic = {}
for i in range(len(su)):
    inset = freq_df.loc[freq_df.support ==su[i]]['itemsets'].apply(lambda x: list(x)).to_list()
    fredic[su[i]] = inset


# In[7]:


cl = []
for index, row in freq_df.iterrows():
    isclose = True
    cli = [x for x in row['itemsets']]
    cls = row['support']
    checkset = fredic[cls]
    
    for i in checkset:
        if (cli!=i):
            if (all(x in i for x in cli)): 
                print(cli, i)
                isclose = False
                break
    
    if(isclose):
        cl.append(cli)   


# In[9]:


closeFP_authors = [", ".join(list(x)) for x in cl]
authorsFP2000 = pd.DataFrame(columns = ["author"])
authorsFP2000["author"] = closeFP_authors
authorsFP2000


# ### It would be helpful to add transaction index to the author patterns

# In[10]:


transaction_index = []
for author in authorsFP2000['author']:
   
    ind = dblp2000['author'].apply(lambda a_list: author in a_list )
    
    transaction_index.append(dblp2000.loc[ind].index.tolist())


# In[11]:


authorsFP2000['transaction_index']  = transaction_index


# In[12]:


authorsFP2000


# In[ ]:


output_path = path
authorsFP2000.to_csv(output_path+"authorsFP2000_with_index.csv", index = False)

