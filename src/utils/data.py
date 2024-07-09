import torch
from src.utils.config import default_config

def synthesize_data(True_Model, dataset_size, save_path, config=default_config):
    """Synthesize data using a given model and save it to a file."""

    Model = True_Model['class']
    new_config = True_Model['new_config']
    config = {**default_config, **new_config} 
    model = Model(config)
    model.eval() # No need to train

    # Generate random data
    X = torch.randn(dataset_size, config['in_dim'])
    Y = model(X)

    # Save the data
    dataset_dict = {'X': X, 'Y': Y}
    torch.save(dataset_dict, save_path)