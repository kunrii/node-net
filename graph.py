import numpy as _lib_
import node as _node_

class Graph:

    def __init__(self):

        self.nodes = set()
        self.inputNodes = set()
        self.outputNodes = set()
        self.links = set()
        self.learning_rate = 1e-3



    ################################################################################################
    #####   TRAINING FUNCTIONALITY FOR EPOCH AND ITERATION
    ################################################################################################



    def train(self, dataset, epochs = 20, batch_size = 64, dataset_size_restriction = None):

        print("\n####################################### TRAINING #######################################\n")

        if (dataset_size_restriction is not None):
            assert dataset_size_restriction >= batch_size

        for n in self.outputNodes:
            if (n.loss == "CROSS_ENTROPY"):
                n.accuracy = list()

        for i in range(epochs):
            self.epoch(dataset, i + 1, batch_size, dataset_size_restriction)
            


    def epoch(self, dataset, epoch_count, batch_size, dataset_size_restriction):

        print("--------------------------------------- epoch {} ---------------------------------------".format(epoch_count))

        Graph.shuffle(dataset)

        if (dataset_size_restriction is None):
            
            dataset_length = dataset["length"]
        else:

            dataset_length = dataset_size_restriction

        for n in self.outputNodes:
            if (n.loss == "CROSS_ENTROPY"):
                n.hit_count = 0

        i = 0
        while (i < dataset_length):

            if (i + batch_size < dataset_length):
                
                self.trainingIteration(dataset, (i, (i + batch_size)), batch_size)
                i += batch_size

            else:
                
                self.trainingIteration(dataset, (i, dataset_length), dataset_length - i)
                break

        for n in self.outputNodes:
            if (n.loss == "CROSS_ENTROPY"):
                n.accuracy.append(n.hit_count / dataset_length * 100)
                print(n.id + " accuracy {}%".format(str(n.accuracy[-1])))



    def trainingIteration(self, dataset, range, batch_size):
        
        self.forwardPropagation(dataset, range, batch_size)
        self.backwardPropagation(batch_size)



    ################################################################################################
    #####   TESTING FUNCTIONALITY
    ################################################################################################



    def test(self, dataset, batch_size = 64):

        print("\n####################################### TESTING #######################################\n")

        dataset_length = dataset["length"]

        print("dataset len " + str(dataset_length))

        #should only be in classification nodes
        for n in self.outputNodes:
            if (n.loss == "CROSS_ENTROPY"):
                n.accuracy = list()
                n.hit_count = 0

        i = 0
        while (i < dataset_length):

            if (i + batch_size < dataset_length):
                
                self.forwardPropagation(dataset, (i, (i + batch_size)), batch_size)
                i += batch_size

            else:
                
                self.forwardPropagation(dataset, (i, dataset_length), dataset_length - i)
                break

        for n in self.outputNodes:
            if (n.loss == "CROSS_ENTROPY"):
                n.accuracy.append(n.hit_count / dataset_length * 100)
                print(n.id + " accuracy {}%".format(n.accuracy[-1]))



    ################################################################################################
    #####   FORWARD PROPAGATION FUNCTIONALITY
    ################################################################################################



    def forwardPropagation(self, dataset, range, batch_size):

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
            self.forwardProcess(n)
    


    def forwardProcess(self, node):

        if (node.process_status == "PROCESSING"):

            raise Exception("Neural network graphs cannot have cycles, but one was detected involving node " + node.id + " with obj id " + str(node))

        elif (node.process_status == "DONE"):

            return

        node.process_status = "PROCESSING"

        inboundLinks = self.getInboundLinks(node)
        for l in inboundLinks:
            
            if (l.prev_node.process_status == "IDLE"):
                self.forwardProcess(l.prev_node)

            l.passToNext()

        node.setActivatedValue()

        if (node in self.outputNodes):
            node.setError() 

        node.process_status = "DONE"



    def loadInputs(self, n, dataset, range):
        
        neuronInputs = dataset["inputs"][n.id]
        n.neurons_net_in[:,:] = neuronInputs[range[0]:range[1],:]



    def loadOutputs(self, n, dataset, range):

        neuronOutputs = dataset["outputs"][n.id]
        n.observations[:,:] = neuronOutputs[range[0]:range[1],:]



    ################################################################################################
    #####   BACKWARD PROPAGATION FUNCTIONALITY
    ################################################################################################
    


    def backwardPropagation(self, batch_size, standard_batch_size = None):

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

        if (node.process_status == "PROCESSING"):

            raise Exception("Neural network graphs cannot have cycles, but one was detected involving node " + node.id + " with obj id " + str(node))

        elif (node.process_status == "DONE"):

            return

        node.process_status = "PROCESSING"

        outboundLinks = self.getOutboundLinks(node)
        for l in outboundLinks:
            
            if (l.next_node.process_status == "IDLE"):
                self.backwardProcess(l.next_node)
            l.passToPrev()

        node.setDelta(outboundLinks)

        node.process_status = "DONE"

        
                
    ################################################################################################
    #####   MISC FUNCTIONALITY
    ################################################################################################
        


    #based on https://stackoverflow.com/questions/35646908/numpy-shuffle-multidimensional-array-by-row-only-keep-column-order-unchanged
    def shuffle(data):

        permutations = _lib_.arange(data["length"])
        _lib_.random.shuffle(permutations)

        for data_dir_key, data_dir_val in data.items():
            if (data_dir_key == "inputs" or data_dir_key == "outputs"):
                for node_key, node_val in data_dir_val.items():
                    node_val[:,:] = node_val[permutations]



    ################################################################################################
    #####   NODE MANAGEMENT FUNCTIONALITY
    ################################################################################################



    def addNode(self, n):
        
        if (n not in self.nodes):
            self.nodes.add(n)
            


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
        if (l not in self.links):
            self.links.add(l)



    def removeLink(self, l):
        if (l in self.links):
            self.links.remove(l)