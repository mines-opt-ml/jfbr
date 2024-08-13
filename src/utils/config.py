default_config = {
    # device
    'preferred_cuda': 0,

    # synthetic data
    'in_dim': 15,
    'out_dim': 15,
    'dataset_size': 1000,
    'train_fraction': 0.8,

    # model
    'max_iter': 100,
    'tol':1e-6,

    # training
    'batch_size': 32,
    'm':1,
    'decay':0.5
}