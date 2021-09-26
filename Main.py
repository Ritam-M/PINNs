import os
from collections import OrderedDict

import torch 
import scipy.io
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

from pylab import meshgrid
from matplotlib import pyplot as plt

from Train import Closure
from Model import PhysicsINN
from preprocess import Preprocessing
from Postprocess import Postprocessing

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

DATA_PATH = "/Data/burgers_shock.mat"
train_x, train_t, train_u = Preprocessing(DATA_PATH)
model = PhysicsINN(4, 40)
n_u, n_f = 100, 10000

for param in model.parameters():
    print(type(param), param.size())
    
EPOCHS = 100
optimizer = torch.optim.LBFGS(model.parameters())

t_bar = trange(EPOCHS)

for epoch in t_bar:
  optimizer.step(Closure)

test_X, test_Y = postprocess(training_points)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(test_Y[0],  linewidth=5)
ax.plot(model(torch.tensor(test_X[0]).float()).detach().numpy(), "r--", linewidth=5)

fig, axs = plt.subplots(1,3, sharey=True, figsize=(20,5))
width = 6

for i,t in enumerate([0.25, 0.50, 0.75]):
    
    axs[i].set_title(f"$U(t,x)$ at t={t}")
    axs[i].plot(test_Y[i],  linewidth=width, label="ground truth")
    axs[i].plot(model(torch.tensor(test_X[i]).float()).detach().numpy(), "r--", linewidth=width, label='prediction')
    axs[i].legend(loc='upper right')

fig.savefig('model_prediction.png')
