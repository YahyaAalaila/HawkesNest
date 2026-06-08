# hawkesnest/datasets/core.py
from __future__ import annotations
import numpy as np, torch
from typing import Sequence, List, Tuple, Optional
from torch.utils.data import Dataset

from hawkesnest.config import SimulatorConfig
import yaml

class StdScaler:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std.clamp_min(1e-9)

    @classmethod
    def fit(cls, stacked: torch.Tensor) -> "StdScaler":
        return cls(stacked.mean(0, keepdim=True),
                   stacked.std(0, keepdim=True))

    def transform(self, x):  return (x - self.mean) / self.std
    def inverse(self,  x):  return x * self.std + self.mean


class SpatioTemporalDataset(Dataset):
    """
    Every item is a T×D tensor  [t, x, y, …].  
    If *scaler* is None and *fit_scaler* is True, we fit on this dataset.
    """
    def __init__(
        self,
        sequences: Sequence[np.ndarray | torch.Tensor],
        scaler: Optional[StdScaler] = None,
        fit_scaler: bool = False,
        normalise: bool = True,
    ):
        self._raw = [torch.as_tensor(s, dtype=torch.float32) for s in sequences]
        self.normalise = normalise

        if normalise:
            if scaler is None and fit_scaler:
                stacked = torch.cat(self._raw, 0)[:, :3]  # t,x,y
                scaler = StdScaler.fit(stacked)
            elif scaler is None:
                raise ValueError(" Provide a fitted scaler or set fit_scaler=True")

        self.scaler = scaler
        self._seqs  = (
            [self._apply_scaler(s) for s in self._raw] if normalise else self._raw
        )
        self._lengths = np.fromiter((len(s) for s in self._seqs), dtype=np.int64)

    # Dataset api 
    def __len__(self): 
        return len(self._seqs)
    
    def __getitem__(self, idx): 
        return self._seqs[idx]

    #  Public helpers 
    def inverse_transform(self, arr: torch.Tensor) -> torch.Tensor:
        if not self.normalise: return arr
        out = arr.clone(); out[:, :3] = self.scaler.inverse(out[:, :3]); return out

    def ordered_indices(self): return np.argsort(self._lengths)

    @staticmethod
    def collate_pad(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        L = torch.tensor([len(s) for s in batch])
        T, D = int(L.max()), batch[0].shape[1]
        out = torch.full((len(batch), T, D), float("nan"))
        for i, s in enumerate(batch): out[i, : len(s)] = s
        return out, L

    # internel helper
    def _apply_scaler(self, seq): 
        seq = seq.clone(); seq[:, :3] = self.scaler.transform(seq[:, :3]); return seq
        
    @classmethod
    def from_simulation(
        cls,
        yaml_str: str,
        n_events: int,
        n_realisations: int = 1,
        seed: int = 0,
        train_frac: float = 1.0,
        normalise: bool = True,
        fit_scaler: bool = True,
    ) -> "SpatioTemporalDataset":
        """
        Build one or more Hawkes simulations from a YAML spec,
        then construct the dataset, optionally splitting into train/test.
        """
        # 1) parse + build simulator
        cfg = SimulatorConfig.model_validate(yaml.safe_load(yaml_str))
        sim = cfg.build()

        # 2) simulate N realisations
        all_seqs = []
        for r in range(n_realisations):
            events, _ = sim.simulate(n=n_events, seed=seed + r)
            arr = events[["t","x","y"]].to_numpy(dtype=np.float32)
            all_seqs.append(arr)

        # 3) if requested, split into train/test
        n_train = int(len(all_seqs) * train_frac)
        train_seqs = all_seqs[:n_train]
        test_seqs  = all_seqs[n_train:]

        # 4) build the “train” dataset
        ds_train = cls(
            sequences=train_seqs,
            scaler=None,
            fit_scaler=fit_scaler if normalise else False,
            normalise=normalise
        )

        # 5) if you also want a test dataset...
        if test_seqs:
            ds_train.test_set = test_seqs   # you can store it for later
        return ds_train
