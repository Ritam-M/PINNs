def preprocess():
    data = scipy.io.loadmat('../input/pinns-data/Data/cylinder_nektar_wake.mat')
    print(data['t'].shape)
    print(data['X_star'].shape)
    print(data['U_star'].shape)
    print(data['p_star'].shape)

    U_star = data['U_star'] # N x 2 x T
    p_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data 
    XX = np.tile(X_star[:, 0:1], (1, T)) # N x T
    YY = np.tile(X_star[:, 1:2], (1, T)) # N x T
    TT = np.tile(t_star, (1, N)).T # N x T

    print(XX.shape)
    print(YY.shape)
    print(TT.shape)

    # Rearrange Data 
    UU = U_star[:, 0, :] # N x T
    VV = U_star[:, 1, :] # N x T
    pp = p_star # N x T

    print(UU.shape)
    print(VV.shape)
    print(pp.shape)

    ## Flattening
    x = XX.flatten()[:, None] # NT x 1
    y = YY.flatten()[:, None] # NT x 1
    t = TT.flatten()[:, None] # NT x 1

    u = UU.flatten()[:, None] # NT x 1
    v = VV.flatten()[:, None] # NT x 1
    p = pp.flatten()[:, None] # NT x 1

    print(x.shape)
    print(y.shape)
    print(t.shape)
    print(u.shape)
    print(v.shape)
    print(p.shape)
