import numpy as np
import csv
import tqdm
from keras.utils import to_categorical

file = r"C:\Users\Kiran\Desktop\Joy\fer2013.csv"

training_sample_count = 28709
validate_sample_count = 3589
testing_sample_count = 3589

total_sample = 35887
training_sample_label = np.empty(shape=(28709,))
validate_sample_label = np.empty(shape=(3589,))
testing_sample_label = np.empty(shape=(3589,))

training_sample_data = np.ndarray(shape=(28709, 2304))
validate_sample_data = np.ndarray(shape=(3589, 2304))
testing_sample_data = np.ndarray(shape=(3589, 2304))

database_populated = False

def one_hot_encoding(label):
    print ("One hot encoding...")
    #print('Shape of data (BEFORE one hot encode): %s' % str(label.shape))
    encoded = to_categorical(label)
    #print('Shape of data (AFTER  one hot encode): %s\n' % str(encoded.shape))
    return encoded

def reshape_to_4D(data):
    print ("Reshaping...")
    data = data.reshape(-1, 48, 48, 1)
    print ("Data shape: ", data.shape)
    return data

def normalise_between_0_and_1(sample_data):
    print ("Normalizing to [0,1]...")
    return reshape_to_4D(sample_data/255)

def normalise_between_minus_1_and_1(sample_data):
    print ("Normalizing to [-1,1]...")
    temp = np.full(sample_data.shape, 128)
    return reshape_to_4D(((sample_data - temp)/128))

def populateNumpyArrays():
    print ("Populating Numpy Arrays...")
    global training_sample_data, training_sample_label
    global testing_sample_data, testing_sample_label
    global validate_sample_data, validate_sample_label
    global database_populated

    print ("Extracting data from .csv file...")

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter = ',')

        training_count = -1
        validate_count = -1
        test_count = -1

        with tqdm.tqdm(total=total_sample, unit='images') as pbar:
            for row in (reader):
                if (row[2] == "Training"):
                    training_count += 1
                    training_sample_label[training_count] = row[0]
                    training_sample_data[training_count] = np.asarray(row[1].split(' '))
                    pbar.set_postfix(extract="Training", refresh=False)
                    #print ("Train: ", training_count)
                if (row[2] == "PublicTest"):
                    validate_count += 1
                    validate_sample_label[validate_count] = row[0]
                    validate_sample_data[validate_count] = np.asarray(row[1].split(' '))
                    pbar.set_postfix(extract="Validating", refresh=False)
                    #print ("Train: ", validate_count)
                if (row[2] == "PrivateTest"):
                    test_count += 1
                    testing_sample_label[test_count] = row[0]
                    testing_sample_data[test_count] = np.asarray(row[1].split(' '))
                    pbar.set_postfix(extract="Testing", refresh=False)
                    #print ("Test: ", test_count)
                pbar.update()
        database_populated = True

        print ("Extraction done...")

def get_training_sample_and_label(mergeDisgustWithAnger):
    global training_sample_label
    global training_sample_data
    print ("Fetching training sample and label...")
    if database_populated:
        if mergeDisgustWithAnger:
            print ("removing...")
            training_sample_label = np.where(training_sample_label==1, 0, training_sample_label)
            unique_elements, counts_elements = np.unique(training_sample_label, return_counts=True)
            print("Frequency of unique values of the said array:")
            print(np.asarray((unique_elements, counts_elements)).astype(np.int32))
        return training_sample_label, training_sample_data
    else:
        print ("Train: Populate the numpy arrays from .csv first... Usage: < dataset.populateNumpyArrays() >")

def get_validating_sample_and_label(mergeDisgustWithAnger):
    global validate_sample_label
    global validate_sample_data
    print ("Fetching Validating sample and label...")
    if database_populated:
        if mergeDisgustWithAnger:
            validate_sample_label = np.where(validate_sample_label==1, 0, validate_sample_label)
        return validate_sample_label, validate_sample_data
    else:
        print ("Validate: Populate the numpy arrays from .csv first... Usage: < dataset.populateNumpyArrays() >")

def get_testing_sample_and_label(mergeDisgustWithAnger):
    global testing_sample_label
    global testing_sample_data
    print ("Fetching testing sample and label...")
    if database_populated:
        if mergeDisgustWithAnger:
            testing_sample_label = np.where(testing_sample_label==1, 0, testing_sample_label)
        return testing_sample_label, testing_sample_data
    else:
        print ("Test: Populate the numpy arrays from .csv first... Usage: < dataset.populateNumpyArrays() >")
