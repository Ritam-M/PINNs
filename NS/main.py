import numpy as np 
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import scipy.io
import torch.optim as optim
import time
from model import PINN
from Preprocess import preprocess

preprocess()
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20]
#layers = [3, 20, 20]

model   = PINN(x_train, y_train,t_train, u_train, v_train,layers_size= layers,  out_size = 2, params_list= None)

optimizer = optim.Adam(params= model.parameters(), lr= 0.1, weight_decay= 0.01)
epochs = 100

for epoch in range(epochs):

    t0 = time.time()

    u_hat, v_hat, p_hat, f_u, f_v = model.net(x_train, y_train, t_train)
    
    loss_ = model.loss(u_train, v_train, u_hat, v_hat, f_u, f_v)
    loss_print  = loss_

    optimizer.zero_grad()   # Clear gradients for the next mini-batches
    loss_.backward()         # Backpropagation, compute gradients
    optimizer.step()

    t1 = time.time()

    print('Epoch %d, Loss= %.10f, lambda_1= %.4f, lambda_2= %.4f Time= %.2f' % (epoch, loss_print, list(model.parameters())[0].item(), list(model.parameters())[1].item(), t1-t0))
                                                                   
