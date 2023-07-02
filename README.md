# node-net

node-net is a framework to create and train neural networks written entirely in python. It is mostly a proof of concept but it will work for learning problems such as those involving the mnist dataset, which it is prepared to load and run by default. It enables setting up nodes that contain neurons and connecting them to one another, allowing for neural network architectures based on directed acyclic graphs. 

It was designed to make it very easy to extend, if you need additional activation functions or loss functions. Currently, there is only one type of loss, cross entropy, and nodes that have loss functions must be terminal nodes in the neural network's graph.

- Automatic cycle detection (cycles in the graph aren't allowed)

- In addition, it implements adaptive moment estimation and L2 regularization as optimizations

- You can use any linear algebra library that has the required operations such as numpy or cupy

- Link to the files here: http://yann.lecun.com/exdb/mnist/

If you find a bug, open up an issue.
