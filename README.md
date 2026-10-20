# Basic Neural Network from Scratch

A small, educational implementation of a fully connected neural network for handwritten digit classification on **MNIST**. The goal is to make the mathematics behind **forward propagation, backpropagation, cross-entropy loss, and gradient descent** visible in straightforward Python code.

The neural network itself is implemented using **NumPy**, without a neural-network training framework or automatic differentiation. **TensorFlow/Keras is used only to download MNIST.**

## What this project demonstrates

- Preparing MNIST images and one-hot-encoded labels
- Computing predictions with matrix multiplication and activation functions
- Measuring classification error with categorical cross-entropy
- Deriving and implementing gradients using the chain rule
- Updating weights and biases with batch gradient descent
- Tracking training loss and classification accuracy

## Network architecture

The model has an input layer, one hidden layer, and an output layer:

```text
MNIST image             Hidden layer             Output layer
28 × 28 pixels  --->    10 ReLU neurons   --->   10 softmax probabilities
784 input values        W1, b1                   W2, b2
```

Each image is flattened to a vector of 784 normalized pixel values. The output represents the predicted probabilities for digits 0 through 9. The current implementation uses 10 hidden units, because the hidden-layer width is set equal to the number of output classes.

## Project structure

```text
basic-neural-network/
├── activation.py        # Activation functions (ReLU and softmax)
├── mnist_dataset.py     # MNIST loading, shuffling, normalization, one-hot encoding
├── model_gd.py          # Model, forward/backward passes, loss, gradient descent
├── test_notebook.ipynb  # Interactive experimentation
└── README.md
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/okkohli/basic-neural-network.git
cd basic-neural-network
```

Install the dependencies:

```bash
python -m pip install numpy tensorflow jupyter
```

Open the example notebook:

```bash
jupyter notebook test_notebook.ipynb
```

The MNIST dataset is downloaded automatically on first use. `model_gd.py` defines the model and training methods; the notebook provides the interactive entry point.

## Mathematical background

### Notation

For a batch of $m$ examples, each **column** is one example:

| Symbol | Meaning | Shape |
|----|----|----|
| $X=A^{[0]}$ | Input images | $784\times m$ |
| $Y$ | One-hot target labels | $10\times m$ |
| $W^{[1]},b^{[1]}$ | Hidden-layer parameters | $10\times784$, $10\times1$ |
| $Z^{[1]},A^{[1]}$ | Hidden pre-activations and activations | $10\times m$ |
| $W^{[2]},b^{[2]}$ | Output-layer parameters | $10\times10$, $10\times1$ |
| $Z^{[2]},A^{[2]}$ | Output logits and probabilities | $10\times m$ |

### 1. Forward propagation

Each layer first computes an affine transformation:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

The hidden layer applies the **rectified linear unit (ReLU)**:

$$
Z^{[1]} = W^{[1]}X+b^{[1]},\qquad
A^{[1]}=\mathrm{ReLU}(Z^{[1]})
$$

$$
\mathrm{ReLU(z)}=\max(0,z)
$$

The output layer produces class scores and applies **softmax**:

$$
Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}
$$

$$
A^{[2]}_{k,i}=\frac{e^{Z^{[2]}_{k,i}}}{\sum_{j=1}^{10}e^{Z^{[2]}_{j,i}}}
$$

The ten probabilities for each image sum to one. The predicted digit is the index of the largest probability.

### 2. Loss function

For multiclass classification with one-hot labels, the model uses **categorical cross-entropy**:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10}
Y_{k,i}\log(A^{[2]}_{k,i})
$$

Training aims to minimize $J$ by changing the weights and biases. In the code, a small positive constant is added inside the logarithm to avoid evaluating $\log(0)$.

### 3. Backpropagation: deriving gradients with the chain rule

Backpropagation is repeated application of the **chain rule**. It tells us how a small change in a weight or bias changes the loss. For clarity, start with **one training example**, with loss $\ell$; afterward, average the resulting gradients over the $m$ examples in a batch.

#### Step 1: Follow the computational dependencies

The forward path is:

$$
X\longrightarrow Z^{[1]}\longrightarrow A^{[1]}
\longrightarrow Z^{[2]}\longrightarrow A^{[2]}\longrightarrow\ell.
$$

For scalar functions $u(v)$ and $v(t)$, the chain rule states:

$$
\frac{d u}{d t}=\frac{d u}{d v}\frac{d v}{d t}.
$$

For neural networks, a neuron can influence the loss along **multiple paths**, so the contributions from those paths must also be **summed**.

#### Step 2: Differentiate the output layer

For one example, let $a^{[2]}=\mathrm{softmax}(z^{[2]})$ and

$$
\ell=-\sum_k y_k\log(a^{[2]}_k).
$$

The softmax Jacobian has entries

$$
\frac{\partial a^{[2]}_j}{\partial z^{[2]}_k}
=a^{[2]}_j(\mathbf{1}_{j=k}-a^{[2]}_k).
$$

Applying the multivariable chain rule gives

$$
\begin{aligned}
\frac{\partial\ell}{\partial z^{[2]}_k}
&=\sum_j\frac{\partial\ell}{\partial a^{[2]}_j}
\frac{\partial a^{[2]}_j}{\partial z^{[2]}_k}\\
&=\sum_j\left(-\frac{y_j}{a^{[2]}_j}\right)
 a^{[2]}_j(\mathbf{1}_{j=k}-a^{[2]}_k)\\
&=-y_k+a^{[2]}_k\sum_j y_j\\
&=a^{[2]}_k-y_k,
\end{aligned}
$$

because one-hot labels satisfy $\sum_j y_j=1$. Thus the **output error signal** is

$$
\boxed{\delta^{[2]}=\frac{\partial\ell}{\partial z^{[2]}}=a^{[2]}-y.}
$$

This simplification is specific to the softmax + cross-entropy combination.

#### Step 3: Differentiate the output weights and biases

Each output pre-activation is

$$
z^{[2]}_k=\sum_j W^{[2]}_{kj}a^{[1]}_j+b^{[2]}_k.
$$

Therefore

$$
\frac{\partial z^{[2]}_k}{\partial W^{[2]}_{kj}}=a^{[1]}_j,
\qquad
\frac{\partial z^{[2]}_k}{\partial b^{[2]}_k}=1.
$$

Using the chain rule again:

$$
\frac{\partial\ell}{\partial W^{[2]}_{kj}}
=\frac{\partial\ell}{\partial z^{[2]}_k}
\frac{\partial z^{[2]}_k}{\partial W^{[2]}_{kj}}
=\delta^{[2]}_k a^{[1]}_j,
$$

$$
\frac{\partial\ell}{\partial b^{[2]}_k}=\delta^{[2]}_k.
$$

In matrix form, for a batch of $m$ examples, where $\Delta^{[2]}=A^{[2]}-Y$ contains the per-example error signals:

$$
\boxed{\frac{\partial J}{\partial W^{[2]}}=
\frac{1}{m}\Delta^{[2]}(A^{[1]})^T,\qquad
\frac{\partial J}{\partial b^{[2]}}=
\frac{1}{m}\sum_{i=1}^m\Delta^{[2]}_{:,i}.}
$$

### 4. Backpropagation: applying the chain rule to the hidden layer

The hidden-layer gradient requires **two separate chain-rule steps**: propagate the loss through the output layer's linear transformation, then through the hidden activation.

#### Step 4: Propagate from output logits to hidden activations

For hidden neuron $j$, its activation $a^{[1]}_j$ affects **every output logit** $z^{[2]}_k$. The multivariable chain rule therefore requires a sum:

$$
\frac{\partial\ell}{\partial a^{[1]}_j}
=\sum_k
\frac{\partial\ell}{\partial z^{[2]}_k}
\frac{\partial z^{[2]}_k}{\partial a^{[1]}_j}.
$$

Since $z^{[2]}_k=\sum_j W^{[2]}_{kj}a^{[1]}_j+b^{[2]}_k$,

$$
\frac{\partial z^{[2]}_k}{\partial a^{[1]}_j}=W^{[2]}_{kj}.
$$

Consequently,

$$
\boxed{\frac{\partial\ell}{\partial a^{[1]}_j}
=\sum_k W^{[2]}_{kj}\delta^{[2]}_k.}
$$

The sum over $k$ is exactly the $j$-th component of a transposed matrix multiplication:

$$
\boxed{\frac{\partial\ell}{\partial a^{[1]}}
=(W^{[2]})^T\delta^{[2]}.}
$$

**Why the transpose?** The forward weight matrix maps hidden activations to output logits. During backpropagation, we collect the influence of all output logits on each hidden activation, reversing the matrix dimensions. This is the transpose, not a matrix inverse.

#### Step 5: Propagate through ReLU

The hidden activation is $a^{[1]}_j=f(z^{[1]}_j)$. Applying the chain rule once more:

$$
\frac{\partial\ell}{\partial z^{[1]}_j}
=\frac{\partial\ell}{\partial a^{[1]}_j}
\frac{\partial a^{[1]}_j}{\partial z^{[1]}_j}
=\left(\sum_k W^{[2]}_{kj}\delta^{[2]}_k\right)f'(z^{[1]}_j).
$$

For ReLU, we use

$$
f'(z)=\begin{cases}1,&z>0,\\0,&z\leq0.\end{cases}
$$

At zero, ReLU is not differentiable; using zero is a conventional subgradient choice. Combining the neuron-wise equations gives

$$
\boxed{\delta^{[1]}=
\left((W^{[2]})^T\delta^{[2]}\right)
\odot f'(z^{[1]}).}
$$

Here $\odot$ means **element-wise multiplication**. If a hidden neuron's ReLU pre-activation is negative, its gradient is zero for that example.

#### Step 6: Differentiate the hidden weights and biases

Finally, $z^{[1]}_j=\sum_r W^{[1]}_{jr}x_r+b^{[1]}_j$, so

$$
\frac{\partial z^{[1]}_j}{\partial W^{[1]}_{jr}}=x_r,
\qquad
\frac{\partial z^{[1]}_j}{\partial b^{[1]}_j}=1.
$$

By the chain rule:

$$
\frac{\partial\ell}{\partial W^{[1]}_{jr}}
=\delta^{[1]}_j x_r,
\qquad
\frac{\partial\ell}{\partial b^{[1]}_j}=\delta^{[1]}_j.
$$

For the full batch, with $\Delta^{[1]}=((W^{[2]})^T\Delta^{[2]})\odot f'(Z^{[1]})$:

$$
\boxed{\frac{\partial J}{\partial W^{[1]}}=
\frac{1}{m}\Delta^{[1]}X^T,\qquad
\frac{\partial J}{\partial b^{[1]}}=
\frac{1}{m}\sum_{i=1}^m\Delta^{[1]}_{:,i}.}
$$

#### Summary: backward pass

For a batch, using per-example error signals $\Delta$ and averaging **once** in the parameter gradients:

$$
\begin{aligned}
\Delta^{[2]} &= A^{[2]}-Y,\\
\nabla_{W^{[2]}}J &= \frac1m\Delta^{[2]}(A^{[1]})^T,\\
\nabla_{b^{[2]}}J &= \frac1m\sum_i\Delta^{[2]}_{:,i},\\
\Delta^{[1]} &= ((W^{[2]})^T\Delta^{[2]})\odot f'(Z^{[1]}),\\
\nabla_{W^{[1]}}J &= \frac1m\Delta^{[1]}X^T,\\
\nabla_{b^{[1]}}J &= \frac1m\sum_i\Delta^{[1]}_{:,i}.
\end{aligned}
$$

The key principle is: **follow the dependencies backward, multiply derivatives along each path, and sum contributions when paths merge.**

### 5. Gradient descent

Once the gradients are known, update each parameter in the direction that reduces the loss:

$$
W^{[l]}\leftarrow W^{[l]}-\alpha\frac{\partial J}{\partial W^{[l]}}
$$

$$
b^{[l]}\leftarrow b^{[l]}-\alpha\frac{\partial J}{\partial b^{[l]}}
$$

Here $\alpha$ is the **learning rate**. The implementation uses **full-batch gradient descent**: each epoch computes gradients using all selected training examples, rather than mini-batches.

### 6. Classification accuracy

The predicted class for example $i$ is

$$
\hat y_i=\mathrm*{argmax}_{k}A^{[2]}_{k,i}
$$

and the accuracy is

$$
\mathrm{Accuracy}=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[\hat y_i=y_i].
$$

Unlike cross-entropy, accuracy is an evaluation metric, not the differentiable training objective.

## How the code maps to the mathematics

| Mathematical operation | Implementation |
|---|---|
| ReLU / softmax | `Activation.relu()` / `Activation.softmax()` |
| $Z^{[1]},A^{[1]},Z^{[2]},A^{[2]}$ | `Model.forward()` |
| $\delta^{[2]},\delta^{[1]},dW,db$ | `Model.backward()` |
| Cross-entropy | `Model.loss()` |
| Parameter updates | `Model.update_parameters()` |
| Epoch loop | `Model.train()` |
| Accuracy | `Model.accuracy()` |

## Dataset and training details

The loader shuffles the training and test splits independently before selecting subsets. Pixel intensities are scaled to $[0,1]$, images are flattened, and labels are converted to one-hot vectors. The default `Dataset` constructor requests 1,000 training and 1,000 test examples; the dataset instance in `model_gd.py` requests 1,000 training and 100 test examples.

The model defaults to a learning rate of `0.01` and `1000` epochs. Training prints the loss and accuracy every 20 epochs. These are **training metrics**; evaluate predictions separately on held-out test data to measure generalization.

## Limitations and possible extensions

This is a learning project rather than a performance-optimized classifier. Useful next steps include:

- Add mini-batch stochastic gradient descent and reshuffle batches each epoch.
- Make the hidden-layer width configurable.
- Use a fixed random seed for reproducible experiments.
- Add numerical gradient checking to validate backpropagation.
- Compare training and test accuracy and plot the loss curve.
- Improve initialization (for example, He initialization for ReLU).
- Implement a numerically stable softmax and log-softmax cross-entropy.

## Learning objective

The main purpose is to show that a neural network is built from a few understandable mathematical operations: **matrix multiplication, nonlinear activation functions, the chain rule, and iterative parameter updates**. Studying the code alongside the equations is intended to make those operations transparent.
