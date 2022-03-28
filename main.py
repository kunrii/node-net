import sys
import datetime
import time
import othernet

import _Graph
import _Node
import _Link
import _Data

import data
import graph
import node
import link

print(sys.executable)

######################################################################################################################

nA = node.Node(28 * 28, id = "in_node")
nB = node.Node(250, activation = "ReLU", id = "node_b") #250 works fine
nC = node.Node(125, activation = "ReLU", id = "node_c") #125 works fine
nD = node.Node(10, activation = "SOFTMAX", loss = "CROSS_ENTROPY", id = "out_node")

lAB = link.Link(nA, nB, "FULLY_CONNECTED")
lBC = link.Link(nB, nC, "FULLY_CONNECTED")
lCD = link.Link(nC, nD, "FULLY_CONNECTED")

nBb = node.Node(nB.neuron_number, id = "node_bias_b")
nBb.asBias()
nCb = node.Node(nC.neuron_number, id = "node_bias_c")
nCb.asBias()
nDb = node.Node(nD.neuron_number, id = "node_bias_out")
nDb.asBias()

lbB = link.Link(nBb, nB, "1_TO_1")
lbC = link.Link(nCb, nC, "1_TO_1")
lbD = link.Link(nDb, nD, "1_TO_1")

neural_net = graph.Graph()
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

train_dataset, test_dataset = data.getMnistData()

neural_net.test(train_dataset)

neural_net.train(train_dataset)

neural_net.test(test_dataset)

# in_val = input("enter to read\n")
# while (in_val != "exit"):
#     image = data.readSingleImage("number.png")
#     neural_net.predict(image)
#     in_val = input("enter to read\n")




