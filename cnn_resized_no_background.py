'''
Directory structure:

data/
    train/
        Type_1/
            t1n1.jpg
            t1n2.jpg
            ...
        Type_2/
            t2n1.jpg
            t2n2.jpg
            ...
        Type_3/
            t2n1.jpg
            t2n2.jpg
            ...
    validation/
        Type_1/
            t1n1.jpg
            t1n2.jpg
            ...
        Type_2/
            t2n1.jpg
            t2n2.jpg
            ...
        Type_3/
            t2n1.jpg
            t2n2.jpg
            ...
    test/
        tn1.jpg
        tn2.jpg
        ...

To run on remote machine (GPU MACHINE) without needing to enter that machine, copy and paste the following three lines into local terminal:

cat cnn_resized_no_background.py | ssh qbit@10.0.1.131 python - outputweightsTEST; scp qbit@10.0.1.131:~/dancrawford/rabbit/outputweightsTEST.h5 .; ssh qbit@10.0.1.131 'rm ~/dancrawford/rabbit/outputweightsTEST.h5'

(you will need to have your rsa public key in qbit@10.0.1.131:~/.ssh/authorized_keys)

'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import sys

output_weights = sys.argv[1]

# dimensions of our images.
img_width, img_height = 100, 100

# train_data_dir = './data/train'
train_data_dir = 'dancrawford/rabbit/data/train'
# validation_data_dir = './data/validation'
validation_data_dir = 'dancrawford/rabbit/data/validation'
nb_train_samples = 1200
nb_validation_samples = 281
epochs = 1
batch_size = 100

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('dancrawford/rabbit/'+output_weights+'.h5')