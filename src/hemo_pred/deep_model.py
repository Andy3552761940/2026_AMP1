from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h).squeeze(-1)


@dataclass
class ESMDeepClassifier:
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    random_state: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.model_: Optional[ResidualMLP] = None
        self.device_ = self.device

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return (x - self.mean_) / self.std_

    def _standardize_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (x - self.mean_) / self.std_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ESMDeepClassifier":
        self._set_seed()
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.device == "cuda" and not torch.cuda.is_available():
            self.device_ = "cpu"
        else:
            self.device_ = self.device

        x_std = self._standardize_fit(x)

        n = len(x_std)
        idx = np.arange(n)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(idx)
        split = max(int(n * 0.85), 1)
        tr_idx = idx[:split]
        va_idx = idx[split:] if split < n else idx[-max(1, n // 5):]

        x_tr = torch.tensor(x_std[tr_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
        x_va = torch.tensor(x_std[va_idx], dtype=torch.float32)
        y_va = torch.tensor(y[va_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=self.batch_size, shuffle=True)

        self.model_ = ResidualMLP(input_dim=x.shape[1], hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device_)

        pos = max(float(y_tr.sum().item()), 1.0)
        neg = max(float(len(y_tr) - y_tr.sum().item()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=self.device_)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_loss = float("inf")
        best_state = None
        stale = 0

        for _ in range(self.max_epochs):
            self.model_.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            self.model_.eval()
            with torch.inference_mode():
                logits_va = self.model_(x_va.to(self.device_))
                val_loss = criterion(logits_va, y_va.to(self.device_)).item()

            if val_loss + 1e-5 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        x = np.asarray(x, dtype=np.float32)
        x_std = self._standardize_transform(x)
        xt = torch.tensor(x_std, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.inference_mode():
            logits = self.model_(xt)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return np.vstack([1.0 - probs, probs]).T
