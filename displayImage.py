import csv
import numpy as np
from matplotlib import pyplot as plt

file = r"C:\Users\Kiran\Desktop\Joy\fer2013.csv"

emotion_map = {'0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy',
           '4': 'Sad', '5': 'Surprise', '6': 'Neutral'}

def plotImage(index):
    with open(file, 'r') as f:
        reader = list(csv.reader(f, delimiter = ','))

        image = np.array(reader[index][1].split())

        image = image.reshape(48, 48).astype(np.int32)

        print (image.shape)

        plt.imshow(image, cmap='gray')

        plt.title(emotion_map[reader[index][0]])

        plt.show()

def plotImagePixels(data, label):
    plt.imshow(data, cmap='gray')

    plt.title(label)

    plt.show()
