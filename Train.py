  import torch 
  
  def closure():
      optimizer.zero_grad()

      output = model(torch.cat((train_t, train_x), dim=1))

      u_grad_x =  torch.autograd.grad(output, train_x, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(output),allow_unused=True)[0]
      u_grad_xx = torch.autograd.grad(u_grad_x, train_x, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(output),allow_unused=True)[0]

      u_grad_t = torch.autograd.grad(output, train_t, retain_graph=True, create_graph=True, grad_outputs=torch.ones_like(output),allow_unused=True)[0]

      f = u_grad_t + output*u_grad_x - (0.01/np.pi) * u_grad_xx

      mse_f = torch.mean(torch.square(f))
      mse_u = torch.mean(torch.square(output - train_u))

      loss = mse_f + mse_u

      loss.backward()

      t_bar.set_description("loss: %.20f" % loss.item())
      t_bar.refresh() # to show immediately the update

      return loss
