import torch
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset
from .dataset import ASPEDDataset
from .sampler import create_weighted_sampler

class ASPEDDataModule(pl.LightningDataModule):
    def __init__(self, train_val_data_list, test_data_list, batch_size=256, 
                 task_type='classification', num_workers=6, n_classes=4):
        super().__init__()
        self.train_val_data_list = train_val_data_list
        self.test_data_list = test_data_list
        self.batch_size = batch_size
        self.task_type = task_type
        self.num_workers = num_workers
        self.n_classes = n_classes

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.test_dataset = ASPEDDataset(self.test_data_list, task_type=self.task_type)

        if stage == 'fit' or stage is None:
            total_size = len(self.train_val_data_list)
            train_size = int(0.9 * total_size)
            val_size = total_size - train_size
            
            # 1. 90/10 Train/Full Val
            train_list, val_list = random_split(
                self.train_val_data_list, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.train_dataset = ASPEDDataset(train_list, task_type=self.task_type)
            
            # 2. Full Validation Set
            self.full_val_dataset = ASPEDDataset(val_list, task_type=self.task_type)
            
            # 3. Small Validation Set
            random.seed(42)
            small_val_size = min(2000, len(val_list))
            small_indices = random.sample(range(len(val_list)), small_val_size)
            self.small_val_dataset = Subset(self.full_val_dataset, small_indices)
            
            if self.task_type == 'classification':
                if self.n_classes == 2:
                    # Use majority label across the 10-sec window for sampling
                    self.train_labels = [
                        1 if sum(1 for l in item['label'] if l > 0) >= 5 else 0
                        for item in train_list
                    ]
                else:
                    # Use the most frequent (mode) label in the window for multiclass sampling
                    def _window_mode(item):
                        labels = [min(int(l), 3) for l in item['label']]
                        return max(set(labels), key=labels.count)
                    self.train_labels = [_window_mode(item) for item in train_list]

    def train_dataloader(self):
        if self.task_type == 'classification':
            sampler = create_weighted_sampler(self.train_labels)
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size,
                sampler=sampler, num_workers=self.num_workers, drop_last=True
            )
        else:
            return DataLoader(
                self.train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers, drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.small_val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def full_val_dataloader(self):
        return DataLoader(
            self.full_val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )