## Tensorflow Important Snippets and Training Pipelines

<h2><b>001. Imports</b></h2>

```
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

import os
import zipfile
import pandas as pd
from glob import glob
import json
import io

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.datasets as tfds

from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Lambda, MaxPooling2D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU, Conv1D , Dropout

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image, ImageFont, ImageDraw



```

<h2><b>002. Hello World with ANNs in Tensorflow</b></h2>

```
y = np.array([3*i+23 for i in range(10000)])
X = np.array([i for i in range(10000)])
```
```
x = Input(shape=(1,))
o = Dense(units=1024, activation='relu')(x)
o = Dense(units=512, activation='relu')(o)
o = Dense(units=256, activation='relu')(o)
o = Dense(units=128, activation='relu')(o)
o = Dense(units=64, activation='relu')(o)
o = Dense(units=1)(o)
model = Model(inputs=x , outputs=o)
model.compile(loss = 'mse' , optimizer = 'adam', metrics=['mean_squared_error'])
```
```
model.summary()

>>
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 1)]               0         
_________________________________________________________________
dense (Dense)                (None, 1024)              2048      
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896     
_________________________________________________________________
dense_4 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 65        
=================================================================
Total params: 699,393
Trainable params: 699,393
Non-trainable params: 0
_________________________________________________________________
```

```
tf.keras.backend.clear_session()
r = model.fit(X, y, epochs=100)
```

```
model.predict([4])
>> array([[34.908173]], dtype=float32)
```

<h2><b>003. Basic Multiclass Singlelabel Image Classification with ANNs</b></h2>

* Let's Analyse the Fashion MNIST Dataset

```
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

```
```
training_images = training_images/255.
test_images = test_images/255.

training_images.shape
>> (60000, 28, 28)
```
* Training The Model
```
i = Input(shape=(28,28,))
o = Flatten()(i)
o = Dense(128, activation='relu')(o) 
o = Dense(10, activation='softmax')(o)

model = Model(inputs = i , outputs = o)

model.compile(loss='sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

models.summary()
>>

Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
```

```
tf.keras.backend.clear_session()
r = model.fit(training_images , training_labels , epochs = 10 , validation_data = (test_images, test_labels))
```

```
plt.plot(r.history['accuracy'], color='red' )
plt.plot(r.history['val_accuracy'], color='blue' )
```

```
plt.plot(r.history['loss'], color='red' )
plt.plot(r.history['val_loss'], color='blue' )
```

* Custom Callback for Early Termination based on Accuracy Reached
```
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.8:
            print("\nStopping training as accuracy reached.. ")
            self.model.stop_training = True
```

```
tf.keras.backend.clear_session()
mycall = CustomCallback()
r = model.fit(training_images , training_labels , epochs = 10 , validation_data = (test_images, test_labels),
              callbacks = [mycall]
             )

```

```
model.predict(test_images[0].reshape(1,28,28))
>> array([[3.2211265e-08, 7.4331405e-13, 9.7391699e-11, 2.9007627e-12,
        7.0998722e-09, 3.1224179e-05, 1.6059880e-09, 2.3650925e-03,
        5.8664429e-09, 9.9760371e-01]], dtype=float32)
```

<h2><b>004. Convolutions and CNNs : Image Classification</b></h2>

```
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nStopping training as accuracy reached.. ")
            self.model.stop_training = True
            
mycall = CustomCallback()

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

i = Input(shape=(28,28,1))
o = Lambda(lambda x: x/255.)(i)

o = Conv2D(32, (3,3) , activation='relu')(o)
o = MaxPooling2D(2,2)(o)
o = Conv2D(32, (3,3) , activation='relu')(o)
o = MaxPooling2D(2,2)(o)

o = Flatten()(o)
o = Dense(128, activation='relu')(o)
o = Dense(10, activation='softmax')(o)

model = Model(inputs = i , outputs = o)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

tf.keras.backend.clear_session()

r = model.fit(training_images , training_labels , epochs = 10 , validation_data = (test_images, test_labels),
              callbacks = [mycall]
             )

```

* We can visualize the Outputs of different Layers using the layers API
```
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

```

<h2><b>005. ImageDataGenerator Usage in Training </b></h2>

```
local_zip = './validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./validation-horse-or-human')

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nStopping training as accuracy reached.. ")
            self.model.stop_training = True
            
mycall = CustomCallback()

train_horse_dir = os.path.join('./horse-or-human/horses')
train_human_dir = os.path.join('./horse-or-human/humans')
validation_horse_dir = os.path.join('./validation-horse-or-human/horses')
validation_human_dir = os.path.join('./validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horse_names = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)


print(len(train_horse_names))
print(len(train_human_names))

batch_size = 8

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        './horse-or-human/',  
        target_size=(300, 300),  
        batch_size=batch_size,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        './validation-horse-or-human/',  
        target_size=(300, 300),  
        batch_size=batch_size,
        class_mode='binary')

i = Input(shape=(300,300,3))
o = Conv2D(16, (3,3) , activation='relu')(i)
o = MaxPooling2D(2,2)(o)
o = Flatten()(o)
o = Dense(128, activation='relu')(o)
o = Dense(1, activation='sigmoid')(o)

model = Model(inputs=i, outputs=o)

print(model.summary())

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

tf.keras.backend.clear_session()
r = model.fit(
    train_generator,
    epochs = 8,
    steps_per_epoch = int((len(train_horse_names) + len(train_human_names))/batch_size),
    validation_data = validation_generator,
    validation_steps = int((len(validation_horse_names) + len(validation_human_names))/batch_size),
    callbacks=[mycall])

```

* Visualize the Activations of the Trained Model

```
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  
x = img_to_array(img) 
x = x.reshape((1,) + x.shape)  

# Scale by 1/255
x /= 255

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of the plot
layer_names = [layer.name for layer in model.layers[1:]]
print(layer_names)


```

```
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:

    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map

    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    
    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
    
      # Tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

```

<h2><b>005. Image Data Augmentation </b></h2>

```
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nStopping training as accuracy reached.. ")
            self.model.stop_training = True
            
mycall = CustomCallback()

train_horse_dir = os.path.join('./datasets/horse-or-human/horses')
train_human_dir = os.path.join('./datasets/horse-or-human/humans')
validation_horse_dir = os.path.join('./datasets/validation-horse-or-human/horses')
validation_human_dir = os.path.join('./datasets/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horse_names = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)


print(len(train_horse_names))
print(len(train_human_names))
```

```
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        './datasets/horse-or-human/',  
        target_size=(300, 300),  
        batch_size=batch_size,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        './datasets/validation-horse-or-human/',  
        target_size=(300, 300),  
        batch_size=batch_size,
        class_mode='binary')

```

```
i = Input(shape=(300,300,3))
o = Conv2D(16, (3,3) , activation='relu')(i)
o = MaxPooling2D(2,2)(o)
o = Flatten()(o)
o = Dense(128, activation='relu')(o)
o = Dense(1, activation='sigmoid')(o)

model = Model(inputs=i, outputs=o)

print(model.summary())

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

tf.keras.backend.clear_session()
r = model.fit(
    train_generator,
    epochs = 8,
    steps_per_epoch = int((len(train_horse_names) + len(train_human_names))/batch_size),
    validation_data = validation_generator,
    validation_steps = int((len(validation_horse_names) + len(validation_human_names))/batch_size),
    callbacks=[mycall])

```

<h2><b>006. Transfer learning : Method 1 </b></h2>
* In this method, we code teh Architecture & Then Load the Downloaded Weights from Disk
* Very Important if we want to do Transfer Learning via Custom Architectures

* Get The Model
```
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 
```

* Training Code
```
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = './datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the base model.
# Set the input shape and remove the dense layers.
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.
for layer in pre_trained_model.layers:
    layer.trainable = False
    
print(pre_trained_model.summary())
```

```
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

# Append the dense network to the base model
model = Model(pre_trained_model.input, x) 

# Print the model summary. See your dense network connected at the end.
model.summary()
```

```
model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

### Rest is same as previous
```

<h2><b>006. Transfer learning : Method 2 </b></h2>

* In this approach, we directly load the model & weights while initializing
* Only possible for models in the tf.keras.applications

```
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as  PretrainedModel, preprocess_input
```

```
acc_thresh = 0.9
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > acc_thresh):
            print("\nStopping training as desired accuracy reached....")
            self.model.stop_training = True
```

```
class TrainingModule(object):
    def __init__(self, root_folder, model_folder, resize_dim = [200,200,3], modelName = "effNetV2B0"):

        self.root_folder = root_folder
        self.IMAGE_SIZE = [200,200]

        self.ptm = PretrainedModel(
        input_shape = self.IMAGE_SIZE + [3],
        weights = 'imagenet',
        include_top = False)

        self.model_folder = model_folder

        self.train_path = os.path.join(root_folder,"train")
        self.test_path = os.path.join(root_folder,"test")

        self.train_image_files = glob(self.train_path + '/*/*')
        self.test_image_files = glob(self.test_path + '/*/*')

        self.folders = glob(self.train_path + "/*")

    def train(self):
        self.ptm.trainable = False
        K = len(self.folders)

        x = Flatten()(self.ptm.output)
        x = Dense(K, activation = 'softmax')(x)

        model = Model(inputs = self.ptm.input , outputs = x)
        
        gen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.2,
        horizontal_flip = True,
        preprocessing_function = preprocess_input
        )

        batch_size = 64

        train_generator = gen.flow_from_directory(
            self.train_path,
            shuffle = True,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        test_generator = gen.flow_from_directory(
            self.test_path,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

        self.model_file = os.path.join(self.model_folder, "trained_model.h5")

        try:
            os.mkdir(self.model_folder)
        except:
            pass

        myCall = myCallback()
        
        r = model.fit(
                train_generator,
                validation_data = test_generator,
                epochs = 8,
                steps_per_epoch = int(np.ceil(len(self.train_image_files)/batch_size)),
                validation_steps = int(np.ceil(len(self.test_image_files)/batch_size)),
                callbacks=[myCall]
        )

        model.save(self.model_file)
        print(f"Saving model at {self.model_file}")
```

```
tm = TrainingModule("./datasets/fullData" , "./datasets/fullData" )

tm.train()
```

<h2><b>006. Image Classification with Multi-Label Multi-Class Output</b></h2>

* Here, everything will remain the same apart from the activation function & loss in the end.
* Use <b>loss = 'binary_crossentropy'</b> instead of <b>loss = 'categorical_crossentropy'</b>
* Use <b>x = Dense(K, activation = 'sigmoid')(x)</b> instead of <b>x = Dense(K, activation = 'softmax')(x)</b>

```
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as  PretrainedModel, preprocess_input


acc_thresh = 0.9
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > acc_thresh):
            print("\nStopping training as desired accuracy reached....")
            self.model.stop_training = True

class TrainingModule(object):
    def __init__(self, root_folder, model_folder, resize_dim = [200,200,3], modelName = "effNetV2B0"):

        self.root_folder = root_folder
        self.IMAGE_SIZE = [200,200]

        self.ptm = PretrainedModel(
        input_shape = self.IMAGE_SIZE + [3],
        weights = 'imagenet',
        include_top = False)

        self.model_folder = model_folder

        self.train_path = os.path.join(root_folder,"train")
        self.test_path = os.path.join(root_folder,"test")

        self.train_image_files = glob(self.train_path + '/*/*')
        self.test_image_files = glob(self.test_path + '/*/*')

        self.folders = glob(self.train_path + "/*")

    def train(self):
        self.ptm.trainable = False
        K = len(self.folders)

        x = Flatten()(self.ptm.output)
        x = Dense(K, activation = 'sigmoid')(x)

        model = Model(inputs = self.ptm.input , outputs = x)
        
        gen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.2,
        horizontal_flip = True,
        preprocessing_function = preprocess_input
        )

        batch_size = 64

        train_generator = gen.flow_from_directory(
            self.train_path,
            shuffle = True,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        test_generator = gen.flow_from_directory(
            self.test_path,
            target_size = self.IMAGE_SIZE,
            batch_size = batch_size
        )

        model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

        self.model_file = os.path.join(self.model_folder, "trained_model.h5")

        try:
            os.mkdir(self.model_folder)
        except:
            pass

        myCall = myCallback()
        
        r = model.fit(
                train_generator,
                validation_data = test_generator,
                epochs = 8,
                steps_per_epoch = int(np.ceil(len(self.train_image_files)/batch_size)),
                validation_steps = int(np.ceil(len(self.test_image_files)/batch_size)),
                callbacks=[myCall]
        )

        model.save(self.model_file)
        print(f"Saving model at {self.model_file}")

tm = TrainingModule("./datasets/fullData" , "./datasets/fullData" )

tm.train()

```

