# Emotion-Recognition-Keras
Emotion Recognition with KERAS.

This repo is for creating a CNN using Keras, that can recognize emotions from images/webcam feed.

KERAS was chosen for it's quick and ease of use.

Trained model is saved in my_model.h5

# Architecture of the CNN implemented
<pre>
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
</pre>
<h3>NOTE:</h3>
<br>
Please refer to test.py/webcam.py for using the trained model on image/webcam feed.
