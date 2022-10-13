#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[28]:


import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# In[42]:


torch.__version__


# In[4]:


data = pd.read_csv('datasets/diabetes.csv')


# In[9]:


data.head()


# In[5]:


x = data.iloc[:,0:-1].values
y_string= list(data.iloc[:,-1])


# In[6]:


y_string


# In[8]:


x.shape


# In[16]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(y_string)
y = le.transform(y_string)


# In[17]:


print(y)


# In[12]:


sc = StandardScaler()
x = sc.fit_transform(x)


# In[18]:


# Now we convert the arrays to PyTorch tensors
x = torch.tensor(x)
# We add an extra dimension to convert this array to 2D
y = torch.tensor(y).unsqueeze(1)


# In[19]:


y.shape


# In[20]:


class Dataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __getitem__(self,index): # override the getitem from the Dataset class
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


# In[21]:


dataset = Dataset(x,y)


# In[27]:


dataset.__getitem__(3)


# In[29]:


# Load the data to your dataloader for batch processing and shuffling
train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=32,
                                           shuffle=True)


# In[31]:


print("There is {} batches in the dataset".format(len(train_loader))) # 768//32 = 24
for (x,y) in train_loader:
    print("For one iteration (batch), there is:")
    print("Data:    {}".format(x.shape))
    print("Labels:  {}".format(y.shape))
    break


# ![demo](https://user-images.githubusercontent.com/30661597/60379583-246e5e80-9a68-11e9-8b7f-a4294234c201.png)

# In[53]:


class Model(nn.Module):
    def __init__(self, input_features, out_features):
        super(Model, self).__init__()    
        
        self.fc1 = nn.Linear(input_features, 5)
        self.fc2 = nn.Linear(5 , 4)
        self.fc3 = nn.Linear(4 , 3)
        self.fc4 = nn.Linear(3, out_features)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


# In[54]:


# Create the network (an object of the Net class)
net = Model(7,1)

#In Binary Cross Entropy: the input and output should have the same shape 
#size_average = True --> the losses are averaged over observations for each minibatch
criterion = nn.BCELoss(reduction = 'mean')   

# We will use SGD with momentum with a learning rate of 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


# In[55]:


net


# In[56]:


epochs = 200


# In[57]:


for ep in range(epochs):
    for inputs,labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        outputs = net(inputs) # net.forward()
        
        loss = criterion(outputs , labels)
        
        # Clear the gradient buffer (we don't want to accumulate gradients)
        optimizer.zero_grad()
        
        # Backpropagation 
        loss.backward()
        
        #Weight Update: w <-- w - lr * gradient
        optimizer.step()
    if ep % 20 == 0:    
        print(f"Epoch is {ep} ...")
        output = (outputs > 0.5).float()
        accuracy = (output == labels).float().mean()
        print(f"Accuracy is {accuracy} , Loss is {loss}")


# In[47]:


a = torch.tensor([1,2,3])
a


# In[48]:


b = torch.tensor([1,2,-3])
b


# In[50]:


(a == b).float()


# In[ ]:




