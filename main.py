import sys

import data as _data_
import graph as _graph_
import node as _node_
import link as _link_

print(sys.executable)                                                               #in case you're running from a virtual environment
sys.setrecursionlimit(2000)                                                         #maximum depth of the neural network, can still crash though

######################################################################################################################

nA = _node_.Node(28 * 28, id = "in_node")
nB = _node_.Node(250, activation = "ReLU", id = "node_b")                             #250 works fine
nC = _node_.Node(125, activation = "ReLU", id = "node_c")                             #125 works fine
nD = _node_.Node(10, activation = "SOFTMAX", loss = "CROSS_ENTROPY", id = "out_node")

lAB = _link_.Link(nA, nB, "FULLY_CONNECTED")
lBC = _link_.Link(nB, nC, "FULLY_CONNECTED")
lCD = _link_.Link(nC, nD, "FULLY_CONNECTED")

nBb = _node_.Node(nB.neuron_number, id = "node_bias_b")
nBb.asBias()
nCb = _node_.Node(nC.neuron_number, id = "node_bias_c")
nCb.asBias()
nDb = _node_.Node(nD.neuron_number, id = "node_bias_out")
nDb.asBias()

lbB = _link_.Link(nBb, nB, "1_TO_1")
lbC = _link_.Link(nCb, nC, "1_TO_1")
lbD = _link_.Link(nDb, nD, "1_TO_1")

neural_net = _graph_.Graph()
neural_net.setInputNode(nA, True)
neural_net.addNode(nB)
neural_net.addNode(nC)
neural_net.setOutputNode(nD, True)
neural_net.addNode(nBb)
neural_net.addNode(nCb)
neural_net.addNode(nDb)

neural_net.addLink(lAB)
neural_net.addLink(lBC)
neural_net.addLink(lCD)
neural_net.addLink(lbB)
neural_net.addLink(lbC)
neural_net.addLink(lbD)

######################################################################################################################

train_dataset, test_dataset = _data_.getMnistData()

neural_net.test(test_dataset)                                                       #expected is 10% with random weights

neural_net.train(train_dataset)

neural_net.test(test_dataset)

neural_net.test(train_dataset)