import csv
import matplotlib.pyplot as plt
import tqdm
import numpy as np

file = r"C:\Users\Kiran\Desktop\Joy\fer2013.csv"

emotion_map = {'0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy',
           '4': 'Sad', '5': 'Surprise', '6': 'Neutral'}

emotion = {'Angry', 'Disgust', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral'}

emotion_without_disgust = {'Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral'}

distribution = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0,
           'Sad': 0, 'Surprise': 0, 'Neutral': 0}

labels = np.empty(shape=(35887,))

with open(file, 'r') as f:
    reader = csv.reader(f, delimiter = ',')

    count = -1
    with tqdm.tqdm(total=35887) as pbar:
        for row in reader:
            count += 1
            labels[count] = row[0]
            distribution[emotion_map[row[0]]] += 1
            pbar.update()

def test_replace():
    global labels
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)).astype(np.int32))

    labels = np.where(labels==1, 0, labels)

    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)).astype(np.int32))

    plt.pie(counts_elements, labels=emotion_without_disgust, autopct='%1.1f%%', shadow=True)

    plt.show()

def visualize():
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    plt.pie(distribution.values(), labels=distribution.keys(), explode=explode, autopct='%1.1f%%', shadow=True)
    plt.show()


test_replace()
#visualize()
