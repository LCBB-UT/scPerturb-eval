import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import issparse


# ============================================================
# PyTorch MLP
# ============================================================

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(128, 64),
                 dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# sklearn-like PyTorch MLP classifier
# ============================================================

class TorchMLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        dropout=0.3,
        alpha=1e-3,
        learning_rate=1e-3,
        batch_size=64,
        max_iter=1000,
        validation_fraction=0.15,
        n_iter_no_change=10,
        tol=1e-4,
        random_state=42,
        device=None
    ):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.classes_ = np.array([0, 1])
        self.n_iter_ = 0

    # def _set_seed(self):
    #     random.seed(self.random_state)
    #     np.random.seed(self.random_state)

    def fit(self, X, y):

        # self._set_seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n_samples = X.shape[0]

        # ----------------------------------------------------
        # Train/Validation split
        # ----------------------------------------------------

        indices = np.arange(n_samples)

        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)

        n_val = int(n_samples * self.validation_fraction)

        if n_val > 0:
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train = X[train_idx]
            y_train = y[train_idx]

            X_val = X[val_idx]
            y_val = y[val_idx]
        else:
            # No validation set: use all data for training
            X_train = X
            y_train = y

            X_val = None
            y_val = None

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = MLPNetwork(
            input_dim=X.shape[1],
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=self.dropout
        ).to(self.device)

        # alpha acts similar to sklearn's L2 regularization
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.alpha
        )

        criterion = nn.BCEWithLogitsLoss()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0

        self.train_losses_ = []
        self.val_losses_ = []

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch in range(self.max_iter):

            self.model.train()

            train_loss_sum = 0.0
            n_train = 0

            for batch_X, batch_y in train_loader:

                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                logits = self.model(batch_X)

                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(batch_X)
                n_train += len(batch_X)

            train_loss = train_loss_sum / n_train

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------
            if n_val > 0:
                self.model.eval()

                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)

                with torch.no_grad():

                    val_logits = self.model(X_val_tensor)

                    val_loss = criterion(
                        val_logits,
                        y_val_tensor
                    ).item()

                self.train_losses_.append(train_loss)
                self.val_losses_.append(val_loss)
                monitored_loss = val_loss
            else:
                # Monitor training loss
                monitored_loss = train_loss

                # Keep val_losses_ aligned with epochs
                self.val_losses_.append(np.nan)

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if monitored_loss < best_loss - self.tol:

                best_loss = monitored_loss

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

            if epochs_without_improvement >= self.n_iter_no_change:
                break

        self.n_iter_ = epoch + 1

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):

        X = np.asarray(X, dtype=np.float32)

        self.model.eval()

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():

            logits = self.model(X_tensor)

            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([
            1 - probabilities,
            probabilities
        ])

    def predict(self, X):

        probabilities = self.predict_proba(X)[:, 1]

        return (probabilities >= 0.5).astype(int)