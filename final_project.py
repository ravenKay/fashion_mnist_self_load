from PIL import Image, ImageFilter
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.image as mpimg #for image uploads
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from scipy import ndimage
from PIL import Image, ImageFilter


im_rows = 28
im_cols = 28

#import Kristina picture test and check the pixels 
test_file = "shoe_test.jpg"
resized_test_file = "resized-"+ test_file

img = Image.open(test_file)
pix = img.load()
np.asarray(pix)
img = img.resize((im_rows,im_cols), Image.ANTIALIAS)
img.save(resized_test_file)
matplotlib.pyplot.imshow(np.asarray(img))
plt.imshow(np.asarray(img))
#plt.show()

img = Image.open(resized_test_file).convert("L")
pix = np.asarray(img)
x_len, y_len = img.size
#line = ""

line = pix/255

#CNN SECTION
#import the two files from kaggle into a pandas data frame

train_df = pd.read_csv(r'fashion-mnist_train.csv')
test_df = pd.read_csv(r'fashion-mnist_test.csv')

print(train_df.head())


# split the training and testing data into X (image) and Y (label) arrays
#min max scaling - normalizing by taking the max-min/spread 
#create computational graph 
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

#rescale all the pixel data so all data is split/divided into labels and pixel data (values are the darkness of the picture 0 to 255)
x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]

#the random state will be 12345
x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.3, random_state=12345,
)

#reshape random image into the random image shape
image_test = x_train[33, :].reshape((28, 28))
#plt.imshow(image_test)
#plt.show()

#Define, compile and fit the model 

im_rows = 28
im_cols = 28 
batch_size = 512 

#add 1 for the third dimension 
im_shape = (im_rows, im_cols, 1)

#unpack tuple and fit into the reshape
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))


#create a sequential model
#filters specifys the output dimenstion of the layer, pass a 3x3 kernel, relu activation function
#maxpooling to downsample the output (instead of 28 will pass 14)
#dropout to drop out random 
# add a few dense layers 
#specify the output layer
#accepts arry of elements, each of which is a layer
cnn_model = Sequential([
        Conv2D(filters=32, kernel_size =(3,3), activation = 'relu', input_shape = im_shape), #first layer needs to know the input shape 
        Dense(32, activation = 'relu'),
        MaxPooling2D(pool_size = 2), 
        Dropout(0.1),
        Flatten(), 
        Dense(32, activation = 'relu'),
        Dense(10, activation = 'softmax') #10 is the number of classes, softmax is predominatly used in the output layer of clustering systems
])

#cnn_model.add(extralayer)

cnn_model.summary()


cnn_model.compile(
        loss = 'sparse_categorical_crossentropy', 
        optimizer = Adam(lr=0.01), 
        metrics = ['accuracy'] #we want to maximize the accuracy
)

cnn_model.fit(
        x_train, y_train, batch_size = batch_size, 
        epochs = 1000, verbose = 1, #verbose determines how much it prints out when training
        validation_data = (x_validate, y_validate),
)

score = cnn_model.evaluate(x_test, y_test, verbose = 0)
#probability = cnn_model.predict_probs(x_test, verbose =0)
print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))


prediction = cnn_model.predict_classes(line.reshape(1,28,28,1))
print(prediction)











