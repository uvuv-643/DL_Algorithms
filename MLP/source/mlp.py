from typing import List, Callable
from tqdm import tqdm
import random 
import numpy as np

class MLP:
  def __init__(
                self,
                x_size: int,
                y_size: int,
                h_sizes: List[int],
                activation_fn: Callable[[float], float],
                activation_fn_prime: Callable[[float], float],
                learning_rate: float
        ):
    self.x = np.zeros(x_size)
    self.y = np.zeros(y_size)
    self.z = [None] * (len(h_sizes) + 1)
    self.a = [None] * (len(h_sizes) + 1)
    self.w = []
    self.b = []
    self.activation_fn = activation_fn
    self.activation_fn_prime = activation_fn_prime
    self.learning_rate = learning_rate

    # init weight matrix, but not values of weights
    self.w.append(np.zeros((h_sizes[0], x_size)))
    self.b.append(np.zeros(h_sizes[0]))
    for i in range(len(h_sizes) - 1):
      self.w.append(np.zeros((h_sizes[i + 1], h_sizes[i])))
      self.b.append(np.zeros(h_sizes[i + 1]))
    self.w.append(np.zeros((y_size, h_sizes[-1])))
    self.b.append(np.zeros(y_size))

    # init weights values
    for i in range(len(self.w)):
      self.w[i][:] = np.random.normal(0, 2 / len(self.w[i][0]), size=self.w[i].shape)
      self.b[i][:] = np.random.normal(0, 2 / len(self.b[i]), size=self.b[i].shape)

  def forward_step(self):
    self.z[0] = self.w[0] @ self.x + self.b[0]
    self.a[0] = self.activation_fn(self.z[0])
    for i in range(1, len(self.w)):
      self.z[i] = self.w[i] @ self.a[i - 1] + self.b[i]
      self.a[i] = self.activation_fn(self.z[i])
    self.y = self.a[-1]

  def backward_step(self, y_actual: List[float]):
    dl_dz = [None] * len(self.w)
    dz_dw = [None] * len(self.w)

    dl_dz[-1] = 2 * (self.y - y_actual) * self.activation_fn_prime(self.z[-1])
    for i in range(len(self.w) - 2, -1, -1):
      dl_dz[i] = self.w[i + 1] * self.activation_fn_prime(self.z[i])

    dz_dw[0] = self.x
    for i in range(1, len(self.w)):
      dz_dw[i] = self.a[i - 1]

    dldz_acc = [np.eye(len(self.y))] # identity matrix
    for i in range(len(self.w) - 1, -1, -1):
      dldz_acc.append(dldz_acc[-1] @ dl_dz[i])
      self.w[i] -= self.learning_rate * np.outer(dldz_acc[-1], dz_dw[i])
      self.b[i] -= self.learning_rate * dldz_acc[-1]

  def fit(self, X, Y, epochs=1):
    for epoch in tqdm(range(epochs)):
      _xy = list(zip(X, Y))
      random.shuffle(_xy)
      for x, y in _xy:
        self.x = x
        self.forward_step()
        self.backward_step(y)

  def predict(self, x):
    self.x = x
    self.forward_step()
    return self.y