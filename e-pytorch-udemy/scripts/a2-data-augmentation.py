#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn


# # Augmentation
# * Makes models robust to changes
# * Helps create more data and reduce data collection time
# * shear, rotation, noise, color distortion, channel reversal, center crop, five crop, saturation , hue , blur , random crop , random resize crop , random hist equalization, Auto augment , Random horizontal flip(p=0.2), Mixup (weighted addition of 2 images and then in labels, use both classes)
# 
# * not all are suitable for every scenario

# ## Mixup Implementation

# In[2]:


from PIL import Image
import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


img_folder = 'images/'
imgs = os.listdir(img_folder)
batch_x = [Image.open(img_folder+p).resize((224,224)) for p in imgs]


# In[6]:


len(batch_x)


# In[ ]:


def normalize_image(x):
    

