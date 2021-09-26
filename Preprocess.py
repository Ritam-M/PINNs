def Preprocessing(DATA_PATH):

    data_dict = scipy.io.loadmat(DATA_PATH)
    n_u, n_f = 100, 10000

    x_data,t_data, u_data = data_dict['x'], data_dict['t'], data_dict['usol']
    x_data.shape, t_data.shape, u_data.shape

    u_t, u_x = meshgrid(t_data, x_data)
    u_t.shape, u_x.shape

    u_data_transformed = u_data.flatten()[:, None] # (25600,1)
    training_points = np.hstack((u_t.flatten()[:, None], u_x.flatten()[:, None])) # (25600, 2)

    IC_X, IC_Y = list(), list() # Initial and boundary points
    CC_X, CC_Y = list(), list() # Collocation points

    for idx, sample in enumerate(training_points):

        t, x = sample
        if t in [0,1] or x in [-1,1]:
            IC_X.append(sample)
            IC_Y.append(u_data_transformed[idx])
        else:
            CC_X.append(sample)
            CC_Y.append(u_data_transformed[idx])

    IC_X = np.array(IC_X)
    IC_Y = np.array(IC_Y)

    CC_X = np.array(CC_X)
    CC_Y = np.array(CC_Y)

    n_u_idx = list(np.random.choice(len(IC_X), n_u))
    n_f_idx = list(np.random.choice(len(CC_X), n_f))

    u_x = torch.tensor(IC_X[n_u_idx, 1:2], requires_grad=True).float()
    u_t = torch.tensor(IC_X[n_u_idx, 0:1], requires_grad=True).float()
    u_u = torch.tensor(IC_Y[n_u_idx, :], requires_grad=True).float()

    f_x = torch.tensor(CC_X[n_f_idx, 1:2], requires_grad=True).float()
    f_t = torch.tensor(CC_X[n_f_idx, 0:1], requires_grad=True).float()
    f_u = torch.tensor(CC_Y[n_f_idx, :], requires_grad=True).float()

    train_x = torch.cat((u_x, f_x), dim=0)
    train_t = torch.cat((u_t, f_t), dim=0)
    train_u = torch.cat((u_u, f_u), dim=0)
    
    return train_x, train_t, train_u
