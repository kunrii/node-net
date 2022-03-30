# NodeNet

NodeNet is a framework to create and train neural networks written entirely in python. It is mostly a proof of concept and was made as a learning exercise, although it will work fine for learning problems such as those involving the mnist dataset, which it is prepared to load and run by default. It was designed to make it very easy to extend, if you need additional activation functions or loss functions. Currently, there is only one type of loss, cross entropy, and nodes that have loss functions must be terminal nodes in the neural network's graph.

In addition, it implements adaptive moment estimation and L2 regularization.

You can use any linear algebra library that has the required operations such as numpy or cupy.

If you find a bug, open up an issue.
