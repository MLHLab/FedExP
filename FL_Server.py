#!/usr/bin/env python
# coding: utf-8

# In[10]:


import socket
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


# # ITERATION 1

# In[11]:


print('***************************** Server Execution Starts ***************************** ')
now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is starting =", current_time)


# In[12]:


# Read model weight from Clinet1
Client1_weight = pickle.load(open('Client1_Europe.pkl', 'rb'))


# In[13]:


# Read model weight from Clinet2
Client2_weight = pickle.load(open('Client2_AsiaPacific.pkl', 'rb'))


# In[14]:


# Read model weight from Clinet3
Client3_weight = pickle.load(open('Client3_America.pkl', 'rb'))


# In[15]:


# Derive model weight from the weights received from the clinets
agg_weight_arr = np.array(Client1_weight) + np.array(Client2_weight) + np.array(Client3_weight)
Central_weight=(1/3)*np.array(agg_weight_arr)


# In[16]:


# Save model weights
pickle.dump(Central_weight, open('Central_Server.pkl', 'wb'))


# In[17]:


now = datetime.now()
current_time = now.strftime("%H:%M:%S.%f")
print("Time when Server Execution is ending =", current_time)
print('***************************** Server Execution Ends ***************************** ')

