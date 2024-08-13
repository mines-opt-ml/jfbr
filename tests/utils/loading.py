from src.utils.loading import load_data

for dataset_type in ['mnist', 'svhn', 'cifar10']:
    train_loader, test_loader = load_data(dataset_type)