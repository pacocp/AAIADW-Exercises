import numpy as np

def f(x):
    return np.sin(x[:,0])*np.cos(x[:,1])+np.sin(0.5*x[:,0])*np.cos(0.5*x[:,1])

def gradient(x):
  grad = np.zeros_like(x)
  grad[:,0] = np.cos(x[:,0])*np.cos(x[:,1]) + 0.5*np.cos(0.5*x[:,0])*np.cos(0.5*x[:,1])
  grad[:,1] = -np.sin(x[:,0])*np.sin(x[:,1]) - 0.5*np.sin(0.5*x[:,0])*np.sin(0.5*x[:,1])
  return grad

def gradient_descent(x0, lr=0.1, n_iters=100):
    points = [x0]
    for i in range(n_iters):
        x0 = x0 - lr * gradient(x0)
        points.append(x0)
    return points

if __name__ == '__main__':
    x = np.random.randint(1, 10, (10,2))
    points = gradient_descent(np.array([[4,4]]), n_iters=5)
    print(points)
