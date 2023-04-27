#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('/data-new2.csv')


# In[2]:


df


# In[4]:


# drop column 1
df = df.drop('Unnamed: 7', axis=1)


# In[5]:


import numpy as np

import tiktoken
import openai


# In[6]:


openai.api_key = "...."


# In[7]:


encoding = tiktoken.get_encoding('cl100k_base')


# In[8]:


df


# In[9]:


#df = df[['S.No','Name','Department', 'Salary', 'Projects', 'Language']]
df.No


# In[10]:


# df['summarised']=('S.No: ' + df.No.str.strip() +'; Name: '+df.Name.str.strip()+ '; Department: '+df.Department.str.strip()+ '; Salary: '+df.Salary.str.strip()+ '; Projects: ' +df.Projects.str.strip()+ '; Language: '+df.Language.str.strip())

 # Concatenate all columns with their heading names
df['summarised'] = df.columns + ': ' + df.astype(str).values.tolist()



# In[11]:


print(df.summarised[1])


# In[12]:


# Concatenate all columns with their heading names as string
df['summarised'] = df.apply(lambda x: '; '.join([str(i)+': '+str(j) for i,j in zip(x.index, x.values)]), axis=1)


# In[13]:


df['tokens'] = df.summarised.apply(lambda x: len(encoding.encode(x)))


# In[14]:


df


# In[15]:


def get_text_embedd(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(
    model=model,
    input=text
    )
#     print("result: ", result['data'][0]['embedding'])
    return result['data'][0]['embedding']


# In[16]:


def get_df_embedding(df):
    return {idx: get_text_embedd(r.summarised) for idx, r in df.iterrows()}


# In[17]:


doc_embedded = get_df_embedding(df)
doc_embedded


# In[18]:


def cal_similar(x, y):
    return np.dot(np.array(x), np.array(y))


# In[19]:


def get_doc_with_similar(query, doc_embedd):
    pass
    query_embedded = get_text_embedd(query)
    doc_similarity = sorted([
        (cal_similar(query_embedded, doc_embedding), doc_index) for doc_index, doc_embedding in doc_embedded.items()
    ], reverse=True)
    return doc_similarity


# In[20]:


get_doc_with_similar("backend?", doc_embedded)


# In[56]:


encoding = tiktoken.get_encoding('gpt2')
seperator_len = len(encoding.encode("\n* "))



def create_prompt(questions, context_embedding, df):
    relevant_document_sections = get_doc_with_similar(questions, context_embedding)
    
    choosen_section=[]
    choosen_section_len=[]
    choosen_section_indexs=[]
    
    for _,section_index in relevant_document_sections:
    
        document_section = df.loc[section_index]
        choosen_section_len += document_section.tokens + seperator_len 
        if choosen_section_len > 500:
            break
            
        choosen_section.append("/n* " + document_section.summarised.replace("/n", " "))
        choosen_section_indexs.append(str(section_index))
        
    header = """ Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the next below, say I am unable to find relevant answer """
#     print(header + "".join(choosen_section) + "\n\n Q: " + questions + "\n A:")
    return header + "".join(choosen_section) + "\n\n Q: " + questions + "\n A:"
        


# In[1]:


def get_answer(query, df, doc_embedded):
   
    prompt = create_prompt(query, doc_embedded, df )
    
    response = openai.Completion.create(
    prompt = prompt,
    temperature = 0,
    max_tokens = 250,
    model = "text-davinci-002", 
    )
#     print("response: ", response)
    return response


# In[ ]:


query= "give me name of backend?"
response = get_answer(query, df, doc_embedded)
print(f"\nQ: {query} \nA: {response['choices'][0]['text']}")

