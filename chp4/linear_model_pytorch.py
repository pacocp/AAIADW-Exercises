from sklearn.datasets import load_iris
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def softmax(x):
    x = x - torch.max(x, dim=1, keepdim=True).values #numerical stability.
    exp = torch.exp(x)
    sum_exp = torch.sum(exp, dim=1, keepdim=True)
    return exp / sum_exp

def initialize_parameters(X, use_bias = True):
    W = torch.randn((X.shape[1], 3), dtype=torch.float32)
    if use_bias:
        b = torch.rand((1, 3), dtype=torch.float32)
        return W, b
    return W, None

def linear_model(X, W, b=None):
    logits = X @ W
    if b != None:
        logits +=  b

    return softmax(logits)
    
def cross_entropy(logits, y):
    return torch.mean(-torch.log(logits[torch.arange(logits.shape[0]), y])) 

def gradient_descent(X,y,X_test, y_test, lr=0.1, n_iters=100, use_bias=False):
    W, b = initialize_parameters(X, use_bias=use_bias)
    losses = []
    acc = []
    y_one_hot = torch.zeros((y.shape[0], 3))
    y_one_hot[torch.arange(y.shape[0]), y] = 1
    for i in tqdm(range(n_iters)):
        logits = linear_model(X, W, b)
        loss = cross_entropy(logits, y)
    
        # Gradient of loss with respect to the logits
        gradient_logits = logits - y_one_hot
    
        # Gradient of loss with respect to W
        gradient_W =  X.T @ gradient_logits

        # Gradient of loss with respect to b
        gradient_b = torch.sum(gradient_logits, dim=0, keepdims=True)
    
        W -= lr * gradient_W
        if use_bias:
            b -= lr * gradient_b

        losses.append(loss.item())
        acc.append(accuracy_score(y, torch.argmax(logits, dim=1)))
 
    logits = linear_model(X_test, W, b)
    acc_test = accuracy_score(y_test, torch.argmax(logits, dim=1))

    return losses, acc, acc_test

data = load_iris()

X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
n_iter = 50
losses, acc, acc_test = gradient_descent(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.int64),
                               torch.tensor(X_test, dtype=torch.float32),
                               torch.tensor(y_test, dtype=torch.int64),
                               lr=0.001,
                               n_iters=n_iter, use_bias=True)

print(f'Acc hold out test set {acc_test}')
fig, axs = plt.subplots(1,2)

axs[0].plot(list(range(n_iter)), losses)
axs[0].set_title('Loss')
axs[1].plot(list(range(n_iter)), acc)
axs[1].set_title('Acc')

plt.show()

