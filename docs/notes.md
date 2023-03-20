# Notes

## Links that I used

1. [PyTorch Deep Learning](https://atcold.github.io/pytorch-Deep-Learning/)
2. [Full Stack Deep Learning](https://atcold.github.io/pytorch-Deep-Learning/)
3. [Dphi Tech](https://bootcamps.dphi.tech/)
4. [Deep Learning with PyTorch](https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans)
5. Other resources such as articles and notebooks will be linked.

---

## 1. Inspiration

[[src](https://atcold.github.io/pytorch-Deep-Learning/en/week01/01-1/)]

Deep learning is inspired by the brain but not all the brain’s details are
relevant.

### Supervised Learning

Most deep learning applications use supervised learning, which is a process of
teaching the machine the correct output by feeding the corresponding inputs.

When the output is correct, we move on. When the output is incorrect, we tweak
the parameter to correct the output toward the desired outcome. The trick is to
figure out which _direction_ and _how much_, this goes back to **gradient
calculation** and **backpropagation**.

### Computing gradients by backpropagation

[How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)

Backpropagation allows you to compute the derivative of the difference of the
output you want and the output you get (which is the value of the objective
function) wrt to any value inside the network. Backpropagation is essential as
it applies to multiple layers.

## 2. Neural Network Architecture

In order for a network to be considered deep, it needs more than 3 layers:

* Input layer
* Hidden layers
* Output layer

Layers can be built using a Sequential model or a Functional model.

* Sequential: a linear stack of layers, layers after layers.
* Functional: building a graph of layers

### Weights

There’s some weight assigned to each connection which represents scalar
multiplication. They are initialized randomly and updated using gradients and
learning rates (implementation can be found below)

### Activation functions

They help activate the neuron if the output from the input reaches a certain
threshold or value, then fire a signal to the next connected neuron(s) in the
next layer(s).

Another way to represent this information would be:

`output = activation_fn(sum(inputs * weights + bias))`

Why non-linearity?

Non-linear functions are those which have degree more than one and they have a

curvature when we plot a non-linear function. Real world data is made up of
non-linear data. If we do not apply an Activation function then the output
signal would simply be a simple linear function.

Some of the popular Activation Functions: _(be careful when calling their names,
especially **relu**)_:

* Sigmoid: `f(x) = 1 / (1 + exp(-x))` between 0 and 1 - for probability
* ReLU: `R(x) = max(0,x)` i.e if x < 0, R(x) = 0 and if x >= 0, R(x) = x,
  between 0 and inf
* tanH: `f(x) = (1 - exp(-2x)) / (1 + exp(-2x))` between -1 and 1
* Leaky ReLU: Soft relu with that instead of squishing gradient to 0, it gives a
  lower value of maybe around 0.01 or so
* Softmax: assigns decimal probabilities to each class in a multi-class problem.
  Those decimal probabilities must add up to 1.0.

Check out this [link](https://machinelearningknowledge.ai/activation-functions-neural-network/).

Some general guidelines:

* sigmoid is usually used in the output layer, especially for binary classification
* at places other than output layer, tanH performs better than sigmoid
* in the hidden layers, if unsure, go for ReLU

### Vanishing and Exploding Gradients

In the case of sigmoid and tanH function, if the weights are large then the gradient will be (vanishingly) small, effectively preventing the weights from changing their values.

Using ReLU or Leaky ReLU is a better approach, they are relatively robust to the vanishing/exploding gradient issue. Leaky ReLU never has 0 gradient, thus they do not disappear and training continues.

### Loss Function

Loss function or sometime error function is used to compute errors between the
predicted values and the target values.

For binary classification, we can use **Binary Cross Entropy Loss**

### Local Minima and Maxima

Functions have hills and valleys, places where they reach a minimum or
maximum value, it can be the local max/min but might not be the global max/
min ofrespective functions.

There is only **one** **global max/min** but there can be multiple **local
max/min**.

Reaching a point in which GD makes very small changes in the cost function is called convergence.

It’s hard to reach global min because we often get stuck in local minimas during training.

### **Gradient Descend**

GD is one of the most popular algorithms to perform optimization and the most common way to optimize neural networks. Some variations:

**Batch GD**: θ = θ − η⋅∇<sub>θ</sub>J(θ)

all training data is taken into consideration to **take a single step**. We take the average of the gradients in all training examples and then use that mean gradient to update our parameters. So that we only take one step per epoch.

* Can be slow, especially in larger dataset

    ```python
    for i in range(nb_epochs):
      params_grad = evaluate_gradient(loss_function, data, params)
      params = params - learning_rate * params_grad
    ```

* For a pre-defined number of epochs, compute gradient _params_grad_ of the
_loss_ _function_ wrt to _params_. Then update the parameters in the opposite
direction of the gradients with the learning rate.
* Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

**SGD (stochastic gradient descent)**: θ = θ − η⋅∇<sub>θ</sub>J(θ;x<sup>(i)</sup>;y<sup>(i)</sup>)

performs a parameter update for **each training example**. It is usually a much
faster technique. It performs one update at a time. Now due to these frequent
updates ,parameters updates have high variance and cause the Loss function to
fluctuate to different intensities.

* ```python
  for i in range(nb_epochs):
      np.random.shuffle(data)
      for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad
  ```

* SGD is relative faster, fits better into memory due to single training sample
  being process
* This is actually a good thing because it helps us discover new and possibly
  better local minima , whereas Standard Gradient Descent will only converge to
  the minimum of the basin as mentioned above.
* This can also cause overshooting, which hinders the convergence.

**Mini-batch GD**: θ = θ − η⋅∇<sub>θ</sub>J(θ;x<sup>(i:i+n)</sup>;y<sup>(i:i+n)</sup>)

performs an update for every batch with n training examples in each batch. It
reduces the variance in the parameter updates , which can ultimately lead us to
a much better and stable convergence. And it also can make use of highly
optimized matrix optimizations.

* ```python
  for i in range(nb_epochs):`
      np.random.shuffle(data)
      for batch in get_batches(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad
  ```

### Challenges

* Choosing a proper learning rate is difficult
* Learning rate schedules try to adjust the learning rate but since they have to
  be defined in advance, which makes it difficult to adapt to a dataset’s
  characteristics
* Since learning rate applies to all parameter updates, if data is sparse and
  features have different frequencies, we do not want to update all of them to
  the same extent.
* Avoiding getting trapped in their numerous suboptimal local minima. These
  saddle points are usually surrounded by a plateau of the same error, which
  makes it notoriously hard for SGD to escape, as the gradient is close to zero
  in all dimensions.

### Optimizers

Optimizers are used to minimize the calculated error by modifying the weights using gradients.

θ = θ−η⋅∇J(θ) — is the formula of the parameter updates, where ‘**η**’ is the learning rate ,’**∇J(θ)**’ is the **Gradient** of **Loss function** - J(θ) w.r.t parameters-‘θ’.

Some popular optimizers such as:

* **Adam (Adaptive Moment Estimation)**: _computes adaptive learning rates for
  each parameter_. In addition to storing an exponentially decaying average of
  past squared gradients like AdaDelta, Adam also keeps an exponentially
  decaying average of past gradients M(t).
* **RMSprop**: _divides the learning rate by an exponentially decaying average
  of squared gradients_.
* **Adagrad: _modifies the general learning rate η at each time step t for every
  parameter θ(i) based on the past gradients that have been computed for θ(i
  _**. It adapts the learning rate to the parameters, performing smaller updates
  for parameters with frequently occurring features and larger updates for
  parameters with infrequent features.
* **AdamW**: fixed weight decay to tackle large weights. In the case of L2
  regularization we add this wd*w to the gradients then compute a moving average
  of the gradients and their squares before using both of them for the update.
  Whereas the weight decay method simply consists in doing the update, then
  subtracting each weight.

**Which one to use?**

* If input data is sparse, adaptive learning rate methods are likely to produce
  better results.
* In summary, RMSprop is an extension of Adagrad that deals with its radically
  diminishing learning rates. Adam, finally, adds bias-correction and momentum
  to RMSprop.

### Batch Normalization

Why normalize the inputs?:

* The weights associated with these inputs would vary a lot because the input
  features present in different ranges, some weights would have to be large and
  then some have to be small.
* Normalize the input features such that all the features would be on the same
  scale, the weights associated with them would also be on the same scale.

Batch normalization:

* In order to bring all the activation values to the same scale, we normalize
  the activation values such that the hidden representation doesn’t vary
  drastically and also helps us to get improvement in the training speed.
* In order to maintain the representative power of the hidden neural network,
  batch normalization introduces two extra parameters — Gamma and Beta.
* The parameters Gamma and Beta are learned along with other parameters of the
  network. If Gamma (γ) is equal to the mean (μ) and Beta (β) is equal to the
  standard deviation(σ) then the activation h_final is equal to the h_norm, thus
  preserving the representative power of the network.

### Dropout

Dropout will deactivate a few neurons in the network randomly to avoid overfitting

* Dropout deactivates the neurons randomly at each training step
* The hidden neurons which are deactivated by dropout changes because of its
  probabilistic behavior

### Review of the working of a NN

1. Forward propagation: feed the inputs to the input layer which will be pass on in the forward direction
2. Gradient descent: minimizing the loss/cost function
3. Backpropagation: traverse back in the network to reduce generated errors by updating weights and biases.

## 3. Classification

Classifying the outcome into classes (binary classification only has 2 classes in the output).

### Binary Classification

Logistic Regression is one of the basic and popular ways to approach the binary classification problem. It outputs a probability that the input belongs to the 2 classes.

Sigmoid function:

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](images/image1.png "image_tooltip")

Take the linear regression function output and pass it through the sigmoid function which outputs the probability between 0 and 1. Class threshold is typically set between 0.5, meaning outputs in range [0.0, 0.5] will belong to class A and range [0.5, 1.0] will be for class B

### **Multiclass Classification**

Classify inputs into more than 2 labels. We typically use softmax function instead of sigmoid for binary classification.

## 4. Convolutional Neural Networks

[[src](https://atcold.github.io/pytorch-Deep-Learning/en/week01/01-2/)]

First, neurons are replicated across the visual field. Second, complex cells that pool information from simple cells (orientation-selective units). As a result, the shift of the picture will change the activation of simple cells, but will not influence the integrated activation of the complex cell (convolutional pooling).

### Deep Learning

Multilayer networks are successful because they exploit the compositional structure of natural data. In compositional hierarchy, combinations of objects at one layer in the hierarchy form the objects at the next layer. If we mimic this hierarchy as multiple layers and let the network learn then we get Deep Learning architecture. Thus, DL networks are hierarchical in nature.

### Feature Extraction

Feature extraction consists of expanding the representational dimension such that the expanded features are more likely to be linearly separable; data points in higher dimensional space are more likely to be linearly separable due to the increase in the number of possible separating planes.

Some common approaches used in feature extraction algorithms:

* Space tiling
* Random projections
* Polynomial Classifiers
* Radial basis functions
* Kernel Machines

Because of the compositional nature of data, learned features have a hierarchy of representations with increasing levels of abstraction. For example:

* Images: images can be thought of as pixels, combinations of pixels constitute edges which when combined form textons (multi-edge shapes). Textons form motifs and motifs from parts of the image.
* Text: there is an inherent hierarchy in textual data. Characters form words, words form word-groups, then clauses, then sentences.
* Speech: samples compose bands, which compose sounds, which compose phones, then phonemes, then whole words then sentences

### Learning representations

SVMs are essentially a very simplistic 2 layer neural net, where the first layer defines “templates” and the second layer is a linear classifier.

## 5. Training and Optimization

### Train, Validation and Test Set

Training dataset: the sample of data to fit the model

Validation dataset: the sample of data used to provide an evaluation of the model’s performance during training or during the process of hyperparameter optimization.

Test dataset: the sample of data for the final evaluation of the model’s performance

### Over/under-fitting

Overfitting: model performs well on training data but poorly on real data or test data, this means that the model has effectively memorized the training data.

Underfitting: model cannot capture the underlying trend of the data, performs poorly on both training and testing data.

Some approaches to avoid overfitting:

* Consider using shallower networks with narrower layers
* Using **Dropouts**: assign specific dropout probability so that the model does not rely heavily on a single neuron or a small group of neurons. It can be used on almost all layers except the output layer.
* Using **Regularization**: slight adjustment to the learning algorithm so the model can generalize better, adding a term into the loss function to penalize large **weights**. **L2 regularization**, etc. We can penalize per specific layers instead of the whole model
* Using Callbacks (Early stopping, Reduce Learning rate): the challenge is to train the network long enough that it is capable of learning the mapping from inputs to outputs but training should not be so long as to overfit the data.

These techniques can be used in combination instead of trying them out one by one.

Notes:

* Large weights in a neural network are a sign of a more complex network that has overfit the training data.
* Probabilistically dropping out nodes in the network is a simple and effective regularization method.
* A large network with more training and the use of a weight constraint are suggested when using dropout.
* Deep networks have better accuracy and vice versa. Adding more layers increases the accuracy but it might lead to Overfitting or generalisation error.
* EarlyStopping basically stops training at the point when performance on a validation dataset starts to degrade. (It would by default need a validation set to be able to work).

## 6. ResNet

In a network with residual blocks, each layer feeds into the next layer and directly into the layers about 2–3 hops away.

### Residual Blocks

Some problems like vanishing gradients and curse of dimensionality, if we have sufficiently deep networks, it may not be able to learn simple functions like an identity function.

R(x) = Output — Input = H(x) — x

H(x) = R(x) + x

The layers in a traditional network are learning the true output (H(x))whereas the layers in a residual network are learning the residual (R(x))

---

## **Practice**

Please find implementations [here](https://github.com/Beomus/AIML-Review). Once the overview is completed, the structure will be revised and links will be updated accordingly.

1. PyTorch
    1. Tensors and Gradients

At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix, or any n-dimensional array.

2. TensorFlow
