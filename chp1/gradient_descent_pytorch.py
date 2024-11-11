import torch

def f(x):
    return torch.sin(x[0])*torch.cos(x[1])+torch.sin(0.5*x[0])*torch.cos(0.5*x[1])

def gradient(x):
  grad = torch.zeros_like(x)
  grad[0] = torch.cos(x[0])*torch.cos(x[1]) + 0.5*torch.cos(0.5*x[0])*torch.cos(0.5*x[1])
  grad[1] = -torch.sin(x[0])*torch.sin(x[1]) - 0.5*torch.sin(0.5*x[0])*torch.sin(0.5*x[1])
  return grad

def gradient_descent(x0, lr=0.1, n_iters=100):
    points = [x0]
    for i in range(n_iters):
        x0 = x0 - lr * gradient(x0)
        points.append(x0)
    return points

if __name__ == '__main__':
    # testing vmap gradient
    x = torch.randint(1, 10, size=(3,2))
    print(x)
    gradient_vmap = torch.vmap(gradient, in_dims=0)
    gradient_result = gradient_vmap(x)
    print(gradient_result)

    start_point = torch.tensor([4,4])
    points = gradient_descent(start_point, n_iters=5)
    print(points)
