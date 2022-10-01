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
import tensorflow_datasets as tfds

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

<h2><b>007. Sequence and Tokens with Tensorflow NLP</b></h2>

```
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

print(word_index)
>> {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}

```

```
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=5)
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)

>>
Word Index =  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

Sequences =  [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]

Padded Sequences:
[[ 0  5  3  2  4]
 [ 0  5  3  2  7]
 [ 0  6  3  2  4]
 [ 9  2  4 10 11]]

Test Sequence =  [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]

Padded Test Sequence: 
[[0 0 0 0 0 5 1 3 2 4]
 [0 0 0 0 0 2 4 1 2 1]]

```

* Test our knowledge on Tokenization
```
with open("./datasets/sarcasm.json", 'r') as f:
    datastore = json.load(f)


sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)

```

<h2><b>008. IMDB Sentiment Analysis with TF</b></h2>

```
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())
    
for s,l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())
    
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```

```
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"
```

```
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
```

```
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])
```

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
```

```
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
```

<h2><b>009. Sarcasm Detection with ANNs and Embedding Layers</b></h2>

```
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open("./datasets/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    
```

```
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
```

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

>>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           160000    
                                                                 
 global_average_pooling1d (G  (None, 16)               0         
 lobalAveragePooling1D)                                          
                                                                 
 dense (Dense)               (None, 24)                408       
                                                                 
 dense_1 (Dense)             (None, 1)                 25        
                                                                 
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0
_________________________________________________________________
```

```
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```

```
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])
```

```
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

```

<h2><b>010. LSTM & GRUs For Sequence Modelling in Texts</b></h2>

```
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))
```

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 10
history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

```
* More Approaches
```
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])
```

```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 50
history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
```


```
# Model Definition with LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```

```
# Model Definition with Conv1D
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

```

<h2><b>011. Text Generation with LSTMs</b></h2>

* We can use different ways.
* This one is a pretty simple one.
* Using a fixed vocabulary, get our LSTM t predict the next words given the previous phrase

```
tokenizer = Tokenizer()
```

```
data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

```

```
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)
```

```
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

```
model = tf.keras.models.Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(xs, ys, epochs=500, verbose=1)
```

* Generating Sentences

```
seed_text = "Ravi went to London"
next_words = 100
  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted = np.argmax(predicted, axis = 1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)

```