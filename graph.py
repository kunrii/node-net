import numpy as _lib_
import node as _node_

class Graph:

    def __init__(self):

        self.nodes = set()
        self.inputNodes = set()
        self.outputNodes = set()
        self.total_loss = list()
        self.links = set()
        self.learning_rate = 1e-3



    ################################################################################################
    #####   TRAINING FUNCTIONALITY FOR EPOCH AND ITERATION
    ################################################################################################



    def train(self, dataset, epochs = 20, batch_size = 64, dataset_size_restriction = None):

        print("\n####################################### TRAINING #######################################\n")

        if (dataset_size_restriction is not None):
            assert dataset_size_restriction >= batch_size

        #should only be in classification nodes
        for n in self.outputNodes:
            n.accuracy = list()

        for i in range(epochs):
            self.epoch(dataset, i + 1, batch_size, dataset_size_restriction)
            



    def epoch(self, dataset, epoch_count, batch_size, dataset_size_restriction):

        print("--------------------------------------- epoch {} ---------------------------------------".format(epoch_count))

        Graph.shuffle(dataset, dataset["length"])

        if (dataset_size_restriction is None):
            
            dataset_length = dataset["length"]
        else:

            dataset_length = dataset_size_restriction

        
        for n in self.outputNodes:
            n.hit_count = 0


        i = 0
        while (i < dataset_length):

            if (i + batch_size < dataset_length):
                
                # self.trainingIteration(epoch_count, i/batch_size + 1, dataset, (i, (i + batch_size)), batch_size, batch_size)
                self.trainingIteration(dataset, (i, (i + batch_size)), batch_size, batch_size)
                i += batch_size

            else:
                
                # self.trainingIteration(epoch_count, i/batch_size + 1, dataset, (i, dataset_length), dataset_length - i, batch_size)
                self.trainingIteration(dataset, (i, dataset_length), dataset_length - i, batch_size)
                break

        for n in self.outputNodes:
            n.accuracy.append(n.hit_count / dataset_length * 100)
            print(n.id + " accuracy {}%".format(str(n.accuracy[-1])))



    def trainingIteration(self, dataset, range, batch_size, standard_batch_size):
        
        # print("------------- epoch {}, iteration {}, this batch size {}, standard batch_size {} ------------- ".format(epoch_count, iteration_count, batch_size, standard_batch_size))

        self.forwardPropagation(dataset, range, batch_size)
        self.backwardPropagation(batch_size)



    #based on https://stackoverflow.com/questions/35646908/numpy-shuffle-multidimensional-array-by-row-only-keep-column-order-unchanged
    def shuffle(data, length):

        permutations = _lib_.arange(length)
        _lib_.random.shuffle(permutations)

        for data_dir_key, data_dir_val in data.items():
            if (data_dir_key == "inputs" or data_dir_key == "outputs"):
                for node_key, node_val in data_dir_val.items():
                    node_val[:,:] = node_val[permutations]



    ################################################################################################
    #####   TESTING FUNCTIONALITY
    ################################################################################################



    ################################################################################################
    #####   PREDICTION FUNCTIONALITY (NO OUTPUT DATA FOR COMPARISON)
    ################################################################################################

    def predict(self, data):

        print("\n####################################### PREDICTING #######################################\n")

        for n in self.nodes:
            n.setNeurons(data["length"])
            n.process_status = "IDLE"

        for n in self.inputNodes:
            self.loadInputs(n, data, (0, data["length"])) 

        for n in self.nodes:
            self.stack_count = -1
            self.forwardProcess(n)

        for n in self.outputNodes:
            for m in range(data["length"]):
                print(n.id + " predicts {}".format(_lib_.argmax(n.neurons_activ, axis = 1)[m]))

    ################################################################################################
    #####   FORWARD PROPAGATION FUNCTIONALITY
    ################################################################################################



    def forwardPropagation(self, dataset, range, batch_size, standard_batch_size = None):

        # print("---------------- forward propagation ----------------")

        #set the neurons to this batch size
        for n in self.nodes:
            n.setNeurons(batch_size)
            n.process_status = "IDLE"

        #load up inputs into the nodes
        for n in self.inputNodes:
            self.loadInputs(n, dataset, range) 

        #load up outputs into the nodes
        for n in self.outputNodes:
            self.loadOutputs(n, dataset, range)

        #forward process the nodes
        for n in self.nodes:
            self.stack_count = -1
            self.forwardProcess(n)
    


    def forwardProcess(self, node):

        self.stack_count += 1
        
        padding = ""
        for i in range(0,self.stack_count):
            padding += "\t"

        if (node.process_status == "PROCESSING"):

            raise Exception("Neural network graphs cannot have cycles, but one was detected involving node " + node.id + " with obj id " + str(node))

        elif (node.process_status == "DONE"):

            return

        node.process_status = "PROCESSING"

        # print(padding + "processing " + node.id + "...")

        #if node has inbound links, make sure the previous nodes are processed and add their contribution to this node's neurons
        inboundLinks = self.getInboundLinks(node)
        for l in inboundLinks:

            # print(padding + "...assessing inbound links from " + l.prev_node.id + " to " + l.next_node.id)
            
            if (l.prev_node.process_status == "IDLE"):
                self.forwardProcess(l.prev_node)
            
            if (l.type == "FULLY_CONNECTED"):
                
                node.neurons_net_in[:,:] += _lib_.matmul(l.prev_node.neurons_activ, l.weights)
            
            elif (l.type == "1_TO_1"):

                node.neurons_net_in[:,:] += l.prev_node.neurons_activ * l.weights.reshape(1,-1)

        #set activation
        node.setActivatedValue(padding)

        #if node has an output term, this needs to be processed
        if (node in self.outputNodes):
            # print(padding + "..." + node.id + " is an output node, calculating error...")
            node.setError(padding + "...") #will also calculate and return the error at some point

        node.process_status = "DONE"

        # print(padding + "..." + node.id + " done processing")

        self.stack_count -= 1



    def loadInputs(self, n, dataset, range):
        # print("loadInputs called")
        neuronInputs = dataset["inputs"][n.id]
        n.neurons_net_in[:,:] = neuronInputs[range[0]:range[1],:]



    def loadOutputs(self, n, dataset, range):

        neuronOutputs = dataset["outputs"][n.id]
        n.observations[:,:] = neuronOutputs[range[0]:range[1],:]



    ################################################################################################
    #####   BACKWARD PROPAGATION FUNCTIONALITY
    ################################################################################################
    
    def backwardPropagation(self, batch_size, standard_batch_size = None):

        # print("---------------- backward propagation ----------------")

        for n in self.nodes:
            n.process_status = "IDLE"

        #backward process the nodes, propagating the deltas
        for n in self.nodes:
            self.stack_count = -1
            self.backwardProcess(n)

        #corrections
        for l in self.links:
            l.correction(self.learning_rate, batch_size)

        

    def backwardProcess(self, node):    

        # pass

        self.stack_count += 1
        
        padding = ""
        for i in range(0,self.stack_count):
            padding += "\t"

        if (node.process_status == "PROCESSING"):

            raise Exception("Neural network graphs cannot have cycles, but one was detected involving node " + node.id + " with obj id " + str(node))

        elif (node.process_status == "DONE"):

            return

        node.process_status = "PROCESSING"

        #only one, and it is the output node
        if (node in self.outputNodes):

            node.delta[:,:] = node.dError.T

        else:

            outboundLinks = self.getOutboundLinks(node)
            for l in outboundLinks:
                
                if (l.next_node.process_status == "IDLE"):
                    self.backwardProcess(l.next_node)

                if (l.type == "FULLY_CONNECTED"):

                    node.delta[:,:] += _lib_.matmul(l.next_node.delta.T, l.weights.T).T #* nodespace.Node.dRelu() in the end

                elif (l.type == "1_TO_1"):

                    #ignore since all of these are bias nodes, delta will enver be needed
                    # node.delta[:,:] += l.next_node.delta * l.weights.reshape(-1, 1)      
                    pass 

            node.delta[:,:] = node.delta[:,:] * _node_.Node.dReLU(node.neurons_net_in).T 

        node.process_status = "DONE"

        #######################################################################################################

        # # print(padding + "processing " + node.id + "...")
        
        # #if node has outbound links, make sure they are processed and add their contribution do this node's delta
        # outboundLinks = self.getOutboundLinks(node)
        # for l in outboundLinks:
            
        #     # print(padding + "...assessing outbound links from " + l.prev_node.id + " to " + l.next_node.id)
        #     if (l.next_node.process_status == "IDLE"):
        #         self.backwardProcess(l.next_node)

        #     if (l.type == "FULLY_CONNECTED"):

        #         node.delta[:,:] += _lib_.matmul(l.next_node.delta.T, l.weights.T).T #* nodespace.Node.dRelu() in the end

        #     elif (l.type == "1_TO_1"):

        #         # print("1_TO_1 link backprop, for node " + node.id)
        #         # print(l.next_node.delta.shape)
        #         # print(l.weights.reshape(1,-1).shape)
        #         # print(node.delta.shape)

        #         node.delta[:,:] += l.next_node.delta * l.weights.reshape(-1, 1)

        # if (node in self.outputNodes):
        #     node.delta[:,:] += node.delta_dError

        # node.delta[:,:] += node.delta * (_node_.Node.dReLU(node.neurons_net_in)).T

        # node.process_status = "DONE"

        # # print(padding + "..." + node.id + " done processing")

        #######################################################################################################

        self.stack_count -= 1
        
                

        




    ################################################################################################
    #####   NODE MANAGEMENT FUNCTIONALITY
    ################################################################################################



    def addNode(self, n):
        
        if (n not in self.nodes):
            
            self.nodes.add(n)
            
            if (n.loss is not None):
                self.outputNodes.add(n)



    def removeNode(self, n):

        if (n in self.nodes):

            if (n in self.inputNodes):
                self.inputNodes.remove(n)
            elif (n in self.outputNodes):
                self.outputNodes.remove(n)

            self.nodes.remove(n)
    


    def setInputNode(self, n, val):
        
        self.setGeneric(n, val, self.inputNodes)



    def setOutputNode(self, n, val):
        
        self.setGeneric(n, val, self.outputNodes)



    def setGeneric(self, n, val, nodeSet):
        
        if (val):
            if (n in self.nodes and n not in nodeSet):  
                nodeSet.add(n)
            elif (n not in self.nodes and n not in nodeSet):
                self.nodes.add(n)
                nodeSet.add(n)
        elif (not val):
            if (n in nodeSet):
                nodeSet.remove(n)       



    def getOutboundLinks(self, node):

        oL = set()

        for l in self.links:
            if (l.prev_node == node):
                oL.add(l)

        return oL



    def getInboundLinks(self, node):

        iL = set()

        for l in self.links:
            if (l.next_node == node):
                iL.add(l)

        return iL



    def addLink(self, l):
        self.links.add(l)



    def removeLink(self, l):
        self.links.remove(l)

