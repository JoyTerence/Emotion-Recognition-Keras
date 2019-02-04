import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping

'''
Following is a convolutional neural network designed to recognize emotions from [48 x 48] image.

Architecture:

Input           -> (1, 48, 48)
Convolution     -> (32, 3, 3)
Convolution     -> (32, 3, 3)
Convolution     -> (32, 3, 3)
MaxPool         -> (2, 2)
Convolution     -> (64, 3, 3)
Convolution     -> (64, 3, 3)
Convolution     -> (64, 3, 3)
MaxPool         -> (2, 2)
Convolution     -> (64, 3, 3)
Convolution     -> (64, 3, 3)
Convolution     -> (64, 3, 3)
MaxPool         -> (2, 2)

---- Flatten to 1D vector ----

Fully Connected -> 256
Fully Connected -> 64
Output layers   -> 6

Activation used -> Relu
Optimizer       -> Adam
Loss Function   -> Categorical Cross Entropy

'''

def CNN_Model(x_train, y_train, x_valid, y_valid):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

    history_summary = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=32, epochs=200, verbose=1, callbacks=[early_stopping_monitor])

    training_accuracy = history_summary.history['acc']
    validation_accuracy = history_summary.history['val_acc']

    model.save('my_model.h5')


    print ("Training Complete!...")
    print ("Training Accuracy = ", training_accuracy)
    print ("Validation Accuracy = ", validation_accuracy)

    del model
