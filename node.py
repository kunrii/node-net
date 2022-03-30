import numpy as _lib_

class Node:

    static_id = -1

    def __init__ (self, neuron_number, activation = None, loss = None, id = None):
        
        self.id =  Node.generateNewID() if id is None else id                       #name your input and output nodes with an id, matching the one in the data. there can be many input and output nodes, so long as they are properly named

        self.neuron_number = neuron_number                                          #the number of neurons
        self.neurons_net_in = None
        self.neurons_activ = None
        
        #nodes can use preloaded values, useful to set bias nodes (load an array of ones), all rows will receive the values of your vector
        self.loaded_value = None                                                    


        self.loss = loss                                                            
        self.observations = None
        self.dError = None

        self.activation = activation

        self.process_status = "IDLE"

        self.delta = None
        self.delta_dError = None

        self.accuracy = None
        self.hit_count = None



    def generateNewID():                                   
        Node.static_id += 1
        return "node_{}".format(Node.static_id)



    #needs to be set at the begining of each train iteration, since they can have odd sized batches
    def setNeurons(self, batch_size):

        if (self.loaded_value is None):                                                 #sets the neuron matrix to 0
            
            self.neurons_net_in = _lib_.zeros((batch_size, self.neuron_number))         #the neurons, dimension 0 is examples from the batch and dimension 1 is neurons
            self.neurons_activ = _lib_.empty((batch_size, self.neuron_number))          #the neurons, dimension 0 is examples from the batch and dimension 1 is neurons
            
            self.delta_dError = _lib_.zeros((self.neuron_number, batch_size))
            self.delta = _lib_.zeros((self.neuron_number, batch_size))
        
        elif (self.loaded_value.shape == (self.neuron_number,)):                        #if you preloaded a value, it will be set at the start of all training iterations to a constant value

            self.neurons_net_in = _lib_.empty((batch_size, self.neuron_number))         #the neurons, dimension 0 is examples from the batch and dimension 1 is neurons
            self.neurons_net_in[:,:] += self.loaded_value.reshape(1,-1)                 #assigns the vector to all of the rows

            self.neurons_activ = _lib_.empty((batch_size, self.neuron_number))

        #only relevant if neuron is an output neuron
        self.observations = _lib_.empty((batch_size, self.neuron_number))
        self.dError = _lib_.empty((batch_size, self.neuron_number))



    #if loaded value is not none, the activation values will be fixed, useful for constant activations like biases
    #function receives a 1d vector
    def setLoadedValue(self, val):
        if (len(val.shape) == 1 and val.shape[0] == self.neurons_net_in):
            self.loaded_value = val



    #shortcut to defining as a node representing a bias
    def asBias(self):
        self.setLoadedValue(_lib_.ones(self.neuron_number))



    def setError(self):

        if (self.loss is None):
            raise Exception("Trying to calculate the error on a node without loss, did you forget the loss or accidentally set it as an output node?")

        if (self.loss == "CROSS_ENTROPY" and self.activation == "SOFTMAX"):

            self.delta_dError[:,:] = (self.neurons_activ - self.observations).T
            self.hit_count += _lib_.count_nonzero(_lib_.argmax(self.neurons_activ, axis = 1) == _lib_.argmax(self.observations, axis = 1))

        else:

            raise Exception("Not implemented")



    def setActivatedValue(self):

        if (self.activation is None):

            self.neurons_activ[:,:] = self.neurons_net_in

        elif (self.activation == "ReLU"):

            self.neurons_activ[:,:] = Node.ReLU(self.neurons_net_in)

        elif (self.activation == "SOFTMAX"):

            self.neurons_activ[:,:] = Node.softmax(self.neurons_net_in)

        else:

            raise Exception("Not implemented")



    def setDelta(self, outboundLinks):

        if (self.activation is None):

            pass

        elif (self.activation == "ReLU"):

            self.delta[:,:] = self.delta * Node.dReLU(self.neurons_net_in).T

        else:

            if (len(outboundLinks) > 0):

                raise Exception("Not implemented for softmax; all nodes implementing softmax must be terminal nodes without any outbound links")

        self.delta += self.delta_dError



    def ReLU(neurons):
        
        return _lib_.maximum(0, neurons)



    def dReLU(neurons):

        return _lib_.heaviside(neurons, 0.5)



    def softmax(neurons):

        temp = neurons - _lib_.max(neurons, axis = 1).reshape(neurons.shape[0],1) #https://stackoverflow.com/questions/42599498/numercially-stable-softmax/42606665#42606665, guards agains potential overflows and does not afect the calculations of the gradient for backpropagation since softmax(X) == softmax(X - c)
        
        return _lib_.exp(temp) / _lib_.sum(_lib_.exp(temp), axis = 1).reshape(temp.shape[0],1) #actual softmax