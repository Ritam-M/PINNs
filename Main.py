model = PhysicsINN(4, 40)

# Flexible HP
# 1. Hidden layers
# 2. Num_neurons
# 3. Activation functions

# Check the layer sizes
for param in model.parameters():
    print(type(param), param.size())
    
EPOCHS = 100
optimizer = torch.optim.LBFGS(model.parameters())

t_bar = trange(EPOCHS)

for epoch in t_bar:
  optimizer.step(evaluate)
