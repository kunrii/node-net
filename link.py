import numpy as _lib_
import node as nodespace

class Link:

    def __init__(self, prev_node, next_node, type):

        self.prev_node = prev_node
        self.next_node = next_node
        
        self.type = type
    
        if (self.type == "FULLY_CONNECTED"):
            
            self.weights = _lib_.random.randn(self.prev_node.neuron_number, self.next_node.neuron_number)
            
            self.raw_mo_1 =_lib_.zeros((self.prev_node.neuron_number, self.next_node.neuron_number)) #estimator for the first raw moment of the gradient of the weights
            self.raw_mo_2 =_lib_.zeros((self.prev_node.neuron_number, self.next_node.neuron_number)) #estimator for the second raw moment of the gradient of the weights

        elif (self.type == "1_TO_1"):

            assert self.prev_node.neuron_number == self.next_node.neuron_number
            self.weights = _lib_.random.randn(self.prev_node.neuron_number)
            
            self.raw_mo_1 =_lib_.zeros((self.prev_node.neuron_number,)) #estimator for the first raw moment of the gradient of the weights
            self.raw_mo_2 =_lib_.zeros((self.prev_node.neuron_number,)) #estimator for the second raw moment of the gradient of the weights

        else:

            raise Exception("Not implemented")



        self.l2_lambda = 1e-4 #1e-4 appears to work fine for this learning problem
        self.adaptive_moment_estimation_beta1 = 0.9
        self.adaptive_moment_estimation_beta2 = 0.999
        self.adaptive_moment_estimation_eps = 1e-8
        self.adaptive_moment_estimation_current_count = 1



    def correction(self, learning_rate, batch_size):

        l2_reg = self.l2_lambda * (2 * self.weights)

        if (self.type == "FULLY_CONNECTED"):

            # print("correction")
            # print(self.next_node.delta.sum())
            # print(self.prev_node.neurons_activ.sum())

            dW = _lib_.matmul(self.next_node.delta, self.prev_node.neurons_activ).T / batch_size

            # print("fully connected delta sum {}".format(_lib_.sum(dW)))

            # self.weights[:,:] += - learning_rate * dW

        elif (self.type == "1_TO_1"):

            dW = _lib_.sum(self.next_node.delta.T * nodespace.Node.dReLU(self.next_node.neurons_net_in), axis = 0) / batch_size

            # print("1 to 1 delta sum {}".format(_lib_.sum(dW)))

            # self.weights[:] += - learning_rate * dW


        self.raw_mo_1 = self.adaptive_moment_estimation_beta1 * self.raw_mo_1 + (1 - self.adaptive_moment_estimation_beta1) * (dW + l2_reg)
        self.raw_mo_2 = self.adaptive_moment_estimation_beta2 * self.raw_mo_2 + (1 - self.adaptive_moment_estimation_beta2) * ((dW + l2_reg) ** 2)
        corrected_raw_moment_1_estimate = self.raw_mo_1 / (1 - self.adaptive_moment_estimation_beta1 ** self.adaptive_moment_estimation_current_count)
        corrected_raw_moment_2_estimate = self.raw_mo_2 / (1 - self.adaptive_moment_estimation_beta2 ** self.adaptive_moment_estimation_current_count)

        corrected_dW = corrected_raw_moment_1_estimate / ((corrected_raw_moment_2_estimate ** 0.5) + self.adaptive_moment_estimation_eps)            

        self.weights += - learning_rate * corrected_dW

        self.adaptive_moment_estimation_current_count += 1

        # print("weight sum after correction {} from ".format(_lib_.sum(self.weights)) + self.prev_node.id + " to " + self.next_node.id)

        

        
