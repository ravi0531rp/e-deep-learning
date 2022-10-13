#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms


# In[2]:


torch.cuda.is_available()


# In[3]:


input_size = 784        #Number of input neurons (image pixels)
hidden_size = 400       #Number of hidden neurons
out_size = 10           #Number of classes (0-9) 
epochs = 10            #How many times we pass our entire dataset into our network 
batch_size = 100        #Input size of the data during one iteration 
learning_rate = 0.001   #How fast we are learning


# In[4]:


train_dataset = datasets.MNIST(root='./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_dataset = datasets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())


# In[12]:


train_dataset


# In[5]:


train_loader = torch.utils.data.DataLoader(
                    dataset = train_dataset,
                    batch_size = batch_size,
                    shuffle = True

)

test_loader = torch.utils.data.DataLoader(
                    dataset = test_dataset,
                    batch_size = batch_size,
                    shuffle = True

)


# In[6]:


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Net, self).__init__()                    
        self.fc1 = nn.Linear(input_size, hidden_size)    #First Layer                           
        self.fc2 = nn.Linear(hidden_size, hidden_size)      #Second Layer Activation
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):                          
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# In[7]:


#Create an object of the class, which represents our network 
net = Net(input_size, hidden_size, out_size)
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# In[14]:


net


# In[8]:


CUDA


# In[9]:


#Train the network
for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):   
        #Flatten the image from size (batch,1,28,28) --> (100,1,28,28) where 1 represents the number of channels (grayscale-->1),
        # to size (100,784) and wrap it in a variable
        images = images.view(-1, 28*28)    
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
            
        outputs = net(images)      # or net.forward(images) 
        _, predicted = torch.max(outputs.data, 1)                                              
        correct_train += (predicted == labels).sum() 
        loss = criterion(outputs, labels)                 # Difference between the actual and predicted (loss function)
        running_loss += loss.item()
        optimizer.zero_grad() 
        loss.backward()                                   # Backpropagation
        optimizer.step()                                  # Update the weights
        
    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
          (epoch+1, epochs, running_loss/len(train_loader), (100*correct_train.double()/len(train_dataset))))
print("DONE TRAINING!")


# In[10]:


with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))


# In[15]:


net.forward(images)


# In[ ]:




