"""
DataLoader Tests
================

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
import torch
from torch.utils.data import TensorDataset
from minigpt.data.dataloader import create_dataloader, InfiniteDataLoader


class TestDataLoader:
    def test_create_dataloader(self):
        x = torch.randn(100, 16)
        y = torch.randn(100, 16)
        ds = TensorDataset(x, y)
        loader = create_dataloader(ds, batch_size=10, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape == (10, 16)

    def test_infinite_dataloader(self):
        x = torch.randn(20, 8)
        ds = TensorDataset(x)
        loader = create_dataloader(ds, batch_size=5, num_workers=0, drop_last=True)
        inf_loader = InfiniteDataLoader(loader)
        # Should cycle past dataset size
        for i, batch in enumerate(inf_loader):
            if i >= 10:
                break
        assert i == 10
