import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def create_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler to handle highly imbalanced datasets.
    It calculates weights inversely proportional to class frequencies, ensuring
    minority classes are oversampled and majority classes are undersampled.

    Args:
        labels (list or numpy.ndarray): The ground truth class labels for the entire training set.
        
    Returns:
        WeightedRandomSampler: A PyTorch sampler to be passed to the DataLoader.
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    
    # Avoid division by zero for classes that might not exist in the current split
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    
    # Assign a weight to each individual sample based on its class
    sample_weights = np.array([class_weights[label] for label in labels])
    
    # Convert to a PyTorch tensor
    sample_weights = torch.from_numpy(sample_weights).double()
    
    # Create the sampler (replacement=True is required for oversampling)
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    return sampler