#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np


# In[2]:


import torch


# In[3]:


import torchvision


# In[4]:


help(torch)


# In[5]:


torch.cuda.is_available()


# # Torch Tensors

# In[6]:


a = torch.tensor([2,2,1])


# In[7]:


a


# In[8]:


type(a)


# In[9]:


b = torch.tensor([
    [2,3,4],
    [4,5,6]    
                 ])


# In[10]:


b


# In[11]:


b.size()


# In[12]:


b.shape


# In[13]:


a.shape


# In[14]:


c = torch.FloatTensor([[2,1,4],[4,7,8]])


# In[15]:


c


# In[16]:


c.dtype


# In[17]:


c.mean()


# In[18]:


c.argmin()


# In[19]:


c.std()


# # Reshape

# In[20]:


b.view(-1,2)


# In[21]:


three_dim = torch.randn(2,3,4)  # channels,rows,columns or c,h,w


# In[22]:


three_dim


# In[23]:


three_dim.shape


# In[24]:


three_dim.view(-1,2)


# In[25]:


three_dim.view(2,-1)


# In[26]:


three_dim.view(2,-1,2)


# In[27]:


torch.rand(2,3)


# In[28]:


torch.randint(6,10,(5,))


# In[29]:


torch.numel(b)


# In[30]:


torch.zeros(3,3,dtype = torch.long)


# In[31]:


torch.ones(3,3,dtype = torch.long)


# In[32]:


t1 = torch.FloatTensor([[2,6,7],[6,8,9]])
t2 = torch.FloatTensor([[-2,-6,-7],[6,8,9]])


# In[33]:


added = torch.add(t1,t2)


# In[34]:


added


# In[35]:


t1.add_(t2)


# In[36]:


t1 # got modified because of that underscore


# In[37]:


t1[:,1]


# In[38]:


t1[1,2]


# In[39]:


t1[1,2].item()


# # Numpy Bridge

# In[40]:


a = torch.ones((2,3))


# In[41]:


b = a.numpy()


# In[42]:


b


# In[43]:


a.add_(1)


# In[44]:


b # Whoa! b changed as well


# In[45]:


a = np.ones((2,3))


# In[46]:


b = torch.from_numpy(a)


# In[47]:


np.add(a,1,out=a)


# In[48]:


b # Whoa! b changed as well


# In[49]:


b = b.cuda()


# In[50]:


b


# # Tensor Concatenation

# In[51]:


f1 = torch.randn(2,5)
f1


# In[52]:


f2 = torch.randn(3,5)
f2


# In[53]:


con_1 = torch.cat([f1,f2] , 0)
con_1


# In[54]:


con_1 = torch.cat([f1,f2] , 1)
con_1


# # Adding and Removing Dimensions to Tensors

# In[55]:


t1 = torch.tensor([1,2,3,4])
t_unsq = torch.unsqueeze(t1,0)


# In[56]:


t_unsq


# In[57]:


t_unsq.squeeze()


# # AutoGrad and Automatic Diferentiation

# In[58]:


x = torch.tensor([1.,2.,3.], requires_grad=True)
y = torch.tensor([11.,24.,-3.], requires_grad=True)

# can use x.requires_grad_() if not already set up previously


# In[59]:


z = x*y+y


# In[60]:


z.grad_fn


# In[61]:


s = z.sum()


# In[62]:


s.grad_fn


# In[63]:


s.backward()


# In[64]:


x.grad # differentiation of s wrt x


# In[65]:


y.grad


# In[66]:


new_z = z.detach()


# In[67]:


print(new_z.grad_fn)


# In[68]:


print(x.requires_grad)


# In[69]:


print((x+10).requires_grad)


# In[70]:


with torch.no_grad(): # this stops grad computation ; good for inference or transfer learning
    print((x+10).requires_grad)


# # Loss Functions

# In[71]:


import torch.nn as nn


# In[72]:


prediction = torch.randn(4,5)


# In[73]:


prediction


# In[74]:


label = torch.randn(4,5)


# In[75]:


label


# In[76]:


mse = nn.MSELoss(reduction='none')
loss = mse(prediction, label)
loss


# In[77]:


mse = nn.MSELoss(reduction='mean')
loss = mse(prediction, label)
loss


# In[78]:


def mseLoss(prediction,labela):
    return ((prediction-label)**2).mean()


# In[79]:


mseLoss(prediction,label)


# ------------------------------------
# -----------------------------------

# In[80]:


label = torch.zeros(4,5).random_(0,2)
label


# In[81]:


sigmoid = nn.Sigmoid()


# In[82]:


bce = nn.BCELoss(reduction='mean')


# In[83]:


bce(sigmoid(prediction) , label)


# In[84]:


bce2 = nn.BCEWithLogitsLoss(reduction='mean')
bce2(prediction, label)


# # Weight Initializations

# In[85]:


layer = nn.Linear(5,5)


# In[86]:


help(layer)


# In[87]:


layer.weight


# In[88]:


layer.bias


# In[89]:


layer.weight.data


# In[90]:


nn.init.uniform_(layer.weight, a=0 , b=3) # see, as we had used a Linear layer, requires_grad=True is set


# In[91]:


nn.init.normal_(layer.weight, mean=0 , std=3) # see, as we had used a Linear layer, requires_grad=True is set


# In[92]:


nn.init.constant_(layer.bias,0) # see, as we had used a Linear layer, requires_grad=True is set


# In[93]:


nn.init.xavier_uniform_(layer.weight,gain=1.1) # see, as we had used a Linear layer, requires_grad=True is set


# In[ ]:




