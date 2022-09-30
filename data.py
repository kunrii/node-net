import numpy as _lib_                           #a linear algebra library, such as numpy or cupy
import struct
from array import array
from os.path import join



# MNIST Data Loader Class
#from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = _lib_.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return x_train, y_train, x_test, y_test       



def getMnistData(normalization = "MIN_MAX"):

    input_path = '.\\dataset'
    training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')    
    
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    train_images, train_labels, test_images, test_data = mnist_dataloader.load_data()

    return getInput(train_images, train_labels, normalization), getInput(test_images, test_data, normalization)



#data is a dictionary containing matrices of input and output values associated with a node identification, as well as a length parameter
#in addition, one hot encoding of labels is done
def getInput(inputs, labels, normalization):

    assert len(inputs) == len(labels)                                       #some validations regarding the dataset size
    length = len(inputs)                                                    

    input_data = _lib_.empty((length, 28 * 28))
    output_data = _lib_.zeros((length, 10))

    ############### SETTING INPUTS

    for i in range(length):
        tmp = _lib_.empty((28,28))
        tmp[:,:] = inputs[i][:]
        input_data[i,:] = tmp.reshape(-1)

    input_data[:,:] = normalize(input_data, normalization)

    ############### ONE-HOT ENCODING OF OUTPUTS

    for i in range(length):
        output_data[i, labels[i]] = 1

    return { "length" : length, "inputs" : { "in_node" : input_data }, "outputs" : { "out_node" : output_data } }



def normalize(data, normalization):
    
    if (normalization == "STANDARD_SCORE"): #use for data drawn from normal distribution

        sample_mean = data.reshape(-1).mean()
        sample_std = data.reshape(-1).std()

        return (data - sample_mean) / sample_std
    
    elif (normalization == "MIN_MAX"):

        min_val = data.reshape(-1).min()
        max_val = data.reshape(-1).max()
        return (data - min_val) / (max_val - min_val)    





