
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


# In[4]:


# The closed Frequent Patterns of authors would be mined by FP growth algorithm in mlxtend lib + closed pattern definition
path = 'E:/CS_Master_Degree_UIUC/CS410_Text_Information_system/Project/Project Submission/CourseProject/Dataset/'
dblp2000 = pd.read_csv(path + "DBLP2000.csv")
dblp2000["author"] = dblp2000.apply(lambda x: x["author"].split(", "), axis = 1) # turn to list of authors for each transaction
dblp2000.head()


# In[6]:


# Applied FPgrowth algorithm in MLXend to find frequent patterns given threshold support
dataset = list(dblp2000["author"])
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
thresh = 4/df.shape[0] # used 4 instead of 10 supports as threshold due to smaller dataset
freq_df = fpgrowth(df, min_support=thresh, use_colnames=True)


# In[7]:


freq_df


# #### Find closed frequent itemset using frequent itemset 

# In[8]:


su = freq_df.support.unique() #all unique support count
#Dictionay storing itemset with same support count key
fredic = {}
for i in range(len(su)):
    inset = freq_df.loc[freq_df.support ==su[i]]['itemsets'].apply(lambda x: list(x)).to_list()
    fredic[su[i]] = inset


# In[9]:


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


# In[10]:


closeFP_authors = [", ".join(list(x)) for x in cl]
authorsFP2000 = pd.DataFrame(closeFP_authors)
authorsFP2000.columns = ["author"]
authorsFP2000


# In[12]:


output_path = path
authorsFP2000.to_csv(output_path+"authorsFP2000.csv", index = False)

