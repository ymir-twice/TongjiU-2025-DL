# 2252708 高宽
# 利用双重Relu网络拟合 sin 函数验证通用近似定理
# 1. 搭建自己的自动梯度框架
# 2. 搭建网络
# 3. 生成数据
# 4. 训练网络
# 5. 验证网络
# 运行代码时，装好环境，然后 python my_reluNet.py 即可, 有比较完整的训练日志与结果可视化

import numpy as np
import matplotlib.pyplot as plt
import logging


class Mul():
    def __init__(self):
        self.mem = {}

    def forward(self, inp, w):
        # inp.shape: (N, num_features)
        # w.shape: (in_dim, out_dim)
        # outp.shape: (N, out_dim)
        self.mem['inp'] = inp
        self.mem['w'] = w
        # print(inp.shape, w.shape)
        outp = inp @ w
        return outp

    def backward(self, grad_outp):
        # grad_outp.shape: (N, out_dim)
        # grad_inp.shape: (N, num_features)
        # grad_w.shape: (in_dim, out_dim)
        grad_inp = np.matmul(grad_outp, self.mem['w'].T)
        grad_w = np.matmul(self.mem['inp'].T, grad_outp)
        return grad_inp, grad_w


class Relu():
    def __init__(self):
        self.mem = {}
        self.params = {}

    def forward(self, inp):
        # inp.shape: (N, num_features)
        # outp.shape: (N, num_features)
        self.mem['inp'] = inp
        outp = np.maximum(inp, 0)
        return outp

    def backward(self, grad_outp):
        # grad_outp.shape: (N, num_features)
        # grad_inp.shape: (N, num_features)
        grad_inp = np.where(self.mem['inp'] > 0, grad_outp, 0)
        return grad_inp


class DenseNet():
    def __init__(self, in_dim, out_dim):
        self.mem = {}
        self.mul = Mul()
        self.w = np.random.randn(in_dim, out_dim)
        self.b = np.random.randn(out_dim)
        self.params = {
            'w': self.w,
            'b': self.b,
        }
        self.grads = {}

    def forward(self, inp):
        # inp.shape: (N, in_dim)
        # w.shape: (in_dim, out_dim)
        # outp.shape: (N, out_dim)
        outp = self.mul.forward(inp, self.w)
        outp = outp + self.b
        return outp

    def backward(self, grad_outp):
        # grad_outp.shape: (N, out_dim)
        # grad_inp.shape: (N, in_dim)
        # grad_w.shape:(in_dim, out_dim)
        grad_inp, grad_w = self.mul.backward(grad_outp)
        grad_b = np.sum(grad_outp, axis=0)
        # print(grad_inp.shape)
        self.grads = {
            'w': grad_w,
            'b': grad_b,
        }
        return grad_inp


class Optimizer():
    def __init__(self, model, lr):
        self.lr = lr
        from collections.abc import Iterable
        assert not isinstance(lr, Iterable)

    def step(self):
        for key in model.net.keys():
            for param in model.net[key].params.keys():
                model.net[key].params[param] -= self.lr * model.net[key].grads[param]


class ReluNet():
    def __init__(self, in_dim, num_hidden, out_dim):
        self.relu = Relu()
        self.dense1 = DenseNet(in_dim, num_hidden)
        self.dense2 = DenseNet(num_hidden, out_dim)

    def forward(self, inp):
        # inp.shape: (N, in_dim)
        # outp.shape: (N, out_dim)
        outp = self.dense1.forward(inp)
        outp = self.relu.forward(outp)
        outp = self.dense2.forward(outp)
        self.net = {
            'dense1': self.dense1,
            'relu': self.relu,
            'dense2': self.dense2,
        }
        return outp

    def backward(self, grad_outp):
        # grad_outp.shape: (N, out_dim)
        # grad_inp.shape: (N, in_dim)
        # grad_w.shape: (in_dim, out_dim)
        for key in reversed(list(self.net.keys())):
            grad_outp = self.net[key].backward(grad_outp)    # 迭代梯度

        return grad_outp


if __name__ == "__main__":
    x = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
    y = (np.sin(x) + np.cos(x)) / np.exp(x) - x ** 4

    lr = 0.0000005               # 因为批大小是1000(一次性投入所有数据),所以学习率要小一点
    n_epoch = 30000
    n_hidden = 400

    model = ReluNet(1, n_hidden, 1)
    optimizer = Optimizer(model, lr)
    losses = []

    for i in range(1, n_epoch + 1):
        y_pred = model.forward(x)
        loss = np.sqrt(np.mean(np.square(y_pred - y)))
        losses.append(loss)
        grad = 2 * (y_pred - y)
        model.backward(grad)
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch {i}, Loss: {loss}')


    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax2 = fig.add_subplot(122)
    ax2.plot(x, y, label='True')
    ax2.plot(x, y_pred, label='Predict')
    ax2.set_title('Fitting sin(x)')
    ax2.legend()
    plt.show()


