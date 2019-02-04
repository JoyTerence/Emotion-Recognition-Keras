import dataset
import numpy as np
import model
import displayImage
#import test

training_label = np.array([28709])
validate_label = np.array([3589])
test_label = np.array([3589])

training_data = np.ndarray(shape=(28709, 2304))
validating_data = np.ndarray(shape=(3589, 2304))
testing_data = np.ndarray(shape=(3589, 2304))

train_data = np.ndarray(shape=(28709, 48, 48, 1))
validate_data = np.ndarray(shape=(3589, 48, 48, 1))
test_data = np.ndarray(shape=(3589, 48, 48, 1))

train_one_hot_label = np.ndarray(shape=(28709, 7))
valid_one_hot_label = np.ndarray(shape=(3589, 7))
test_one_hot_label = np.ndarray(shape=(3589, 7))

removeDisgust = True

def setupDataset():
    global training_data, training_label
    global validate_data, validate_label
    global test_data, test_label
    global train_data, train_one_hot_label
    global validate_data, valid_one_hot_label
    global test_data, test_one_hot_label

    print ("Setting up dataset... ")
    dataset.populateNumpyArrays()
    training_label, training_data = dataset.get_training_sample_and_label(removeDisgust)
    validate_label, validating_data = dataset.get_validating_sample_and_label(removeDisgust)
    test_label, testing_data = dataset.get_testing_sample_and_label(removeDisgust)

    #train_data = dataset.normalise_between_0_and_1(training_data)
    #validate_data = dataset.normalise_between_0_and_1(validating_data)
    #test_data = dataset.normalise_between_0_and_1(testing_data)

    train_data = training_data.reshape(28709, 48, 48, 1)
    valid_data = validating_data.reshape(3589, 48, 48, 1)
    test_data = testing_data.reshape(3589, 48, 48, 1)

    train_one_hot_label = dataset.one_hot_encoding(training_label)
    valid_one_hot_label = dataset.one_hot_encoding(validate_label)
    test_one_hot_label = dataset.one_hot_encoding(test_label)

    print ("Setting up done...")

if __name__ == '__main__':
    setupDataset()

    #displayImage.plotImagePixels(train_data[0, :, :].reshape(48, 48), training_label[0])

    model.CNN_Model(train_data, train_one_hot_label, validate_data, valid_one_hot_label)

    #test.predict(test_data[1, :, :, :].reshape(1, 48, 48, 1), test_label[1])
    #displayImage.plotImage(28709+3589+1)
