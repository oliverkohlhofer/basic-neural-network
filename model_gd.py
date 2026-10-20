from py_compile import main

import numpy as np

from activation import Activation
from mnist_dataset import Dataset

Data = Dataset(n_train=1000, n_test=100)

class Model:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None  # Initialize weights and biases to None 
        self.input_dim = None
        self.output_dim = None


    def parameter_init(self):
        W1 = np.random.rand(self.output_dim, self.input_dim) -0.5
        b1 = np.random.rand(self.output_dim, 1) -0.5
        W2 = np.random.rand(self.output_dim, self.output_dim) -0.5
        b2 = np.random.rand(self.output_dim, 1) -0.5
        return W1, b1, W2, b2

    def forward(self, X, W1, b1, W2, b2):
        Z1 = np.dot(W1, X) + b1
        A1 = Activation.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = Activation.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2, W2):
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = 1 / m * np.dot(dZ2, A1.T) 
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2.T.dot(dZ2) * (Z1 > 0)  # (Z > 1) -> Derivative of ReLU g'(Z)
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        return W1, b1, W2, b2

    def loss(self, Y, A2):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A2 + 1e-8)) / m  # Adding a small value to avoid log(0)
        return loss

    def accuracy(self, Y, A2):
        predictions = np.argmax(A2, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels)
        return accuracy

    def train(self, X, Y, track_data:bool=False, Testset:tuple=None):
        train_loss_data = []
        train_acc_data = []
        test_loss_data = []
        test_acc_data = []
        self.input_dim = X.shape[0]
        self.output_dim = Y.shape[0]
        print(f"Input dim: {self.input_dim}, Output dim: {self.output_dim}")
        W1, b1, W2, b2 = self.parameter_init()
        for epoch in range(self.epochs):
            Z1, A1, Z2, A2 = self.forward(X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = self.backward(X, Y, Z1, A1, Z2, A2, W2)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, self.learning_rate)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {self.loss(Y, A2):.4f}, Accuracy = {self.accuracy(Y, A2):.4f}")

            # OPTIONAL: tracking data for process analysis:
            if track_data:
                train_loss_data.append(self.loss(Y,A2))
                train_acc_data.append(self.accuracy(Y,A2))
                if Testset is not None:
                    try:
                        _, _, _, A2_test = self.forward(Testset[0], W1, b1, W2, b2)
                        test_acc_data.append(self.accuracy(Testset[1], A2_test))
                        test_loss_data.append(self.loss(Testset[1], A2_test))
                    except:
                        print("An exception occurred at Test-Set")
                        Testset = None
                        
        # Store the trained weights and biases:             
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

        if track_data:
            return train_loss_data, train_acc_data, test_loss_data, test_acc_data
        else:
            return

    def test(self, X_test, Y_test):
        _, _, _, A2 = self.forward(X_test, self.W1, self.b1, self.W2, self.b2)
        return self.accuracy(Y_test, A2)

    def predict(self, image):
        # Expects a 28x28 image already normalized to [0, 1].
        X = np.asarray(image).reshape(784, 1)
        _, _, _, A2 = self.forward(X, self.W1, self.b1, self.W2, self.b2)
        digit = int(np.argmax(A2[:, 0]))
        return (digit, A2[:, 0])

    
    
