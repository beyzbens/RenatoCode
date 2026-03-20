"""
Ensemble Probabilistic Building Thermal Surrogate Model.

=== MATHEMATICAL FOUNDATION ===

The building's thermal dynamics can be expressed as:
    S(t+1) = S(t) + f(S(t), A(t), W(t))

where:
    S(t) = state vector (zone temperatures, tank temperatures)
    A(t) = action vector (setpoints)
    W(t) = weather (outdoor temperature, solar, humidity)
    f    = thermal transition function (unknown, complex)

EnergyPlus computes f exactly but takes minutes per simulation.
We learn an approximation f_θ using neural networks, enabling
millisecond predictions for rapid flexibility analysis.

=== WHY ENSEMBLE? ===

A single neural network gives point estimates:
    f_θ(S, A) → ΔS_predicted

An ensemble of N independently trained networks gives:
    f_θ₁(S, A) → ΔS₁
    f_θ₂(S, A) → ΔS₂
    ...
    f_θₙ(S, A) → ΔSₙ

From these we compute:
    Mean prediction:  μ = (1/N) Σᵢ ΔSᵢ
    Uncertainty:      σ = std(ΔS₁, ..., ΔSₙ)   (epistemic uncertainty)

Each member also outputs aleatoric (data) uncertainty via
a heteroscedastic Gaussian:
    f_θᵢ(S, A) → (μᵢ, σ²ᵢ)

Total uncertainty combines both:
    σ²_total = (1/N)Σᵢ σ²ᵢ + (1/N)Σᵢ (μᵢ - μ)²
                ─────────────   ──────────────────
                aleatoric        epistemic

=== WHY PROBABILISTIC (Gaussian) OUTPUT? ===

Standard MSE loss: L = (y - ŷ)²
  → Treats all predictions equally, ignores varying difficulty

Gaussian negative log-likelihood:
    L = (1/2) * [(y - μ)² / σ² + log(σ²)]
    
  → Network learns WHEN it's uncertain (large σ²) vs confident (small σ²)
  → Noisy data points get large σ², don't dominate training
  → This is the aleatoric uncertainty

=== PARAMETER CHOICES ===

ensemble_size = 5:
    Literature standard (Chua et al., 2018 "PETS"). 5 provides good
    diversity vs computational cost tradeoff. 3 is minimum, 7+ gives
    dimishing returns.

hidden_dim = 256, 3 layers:
    Building thermal dynamics are relatively smooth functions, but 
    power consumption involves sharp, non-linear step functions.
    Increased to 256 to better capture HVAC on/off triggers.

LayerNorm + SiLU activation:
    LayerNorm: stabilizes training, especially with varying input scales
    (temperatures ~20°C vs power ~5000W).
    SiLU (Sigmoid Linear Unit): smooth, non-monotonic activation,
    better gradient flow than ReLU for regression tasks.

Learning rate = 1e-3 with Adam:
    Standard for regression tasks. Adam handles different parameter
    scales well. Weight decay 1e-5 prevents overfitting.

Bootstrap sampling:
    Each ensemble member sees a different random subset (with replacement)
    of the training data. This is the source of epistemic diversity.
    Without this, all members would converge to the same solution.

max_logvar / min_logvar bounds:
    Prevents the variance output from collapsing to zero (overconfident)
    or exploding to infinity (trivially ignoring all data).
    Implemented as learnable parameters with softplus constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class ProbabilisticMLP(nn.Module):
    """Single ensemble member: learns μ(x) and σ²(x) of state delta.

    Architecture:
        Input(17) → Linear(256) → LayerNorm → SiLU
                  → Linear(256) → LayerNorm → SiLU
                  → Linear(256) → LayerNorm → SiLU
                  ┬→ mean_head(8)      → μ
                  └→ logvar_head(8)    → log(σ²), bounded by [min, max]
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

        # Learnable bounds on log-variance to prevent collapse/explosion
        self.max_logvar = nn.Parameter(torch.ones(1, output_dim) * 0.5)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_dim) * 10.0)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        # Soft clamp: differentiable bounding of log-variance
        # Uses softplus to smoothly enforce: min_logvar ≤ logvar ≤ max_logvar
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class EnsembleThermalSurrogate:
    """Ensemble of probabilistic MLPs for building thermal prediction.

    === INPUT FEATURES (17 values) ===

    State features (13):
        [0] outdoor_T        : outdoor dry-bulb temperature (°C)
        [1] zone_T_living    : living room air temperature (°C)
        [2] zone_T_kitchen   : kitchen air temperature (°C)
        [3] zone_T_bedroom   : bedroom air temperature (°C)
        [4] zone_T_bathroom  : bathroom air temperature (°C)
        [5] buffer_T         : buffer tank water temperature (°C)
        [6] dhw_T            : DHW tank water temperature (°C)
        [7] hour_sin         : sin(2π·hour/24) - cyclical time encoding
        [8] hour_cos         : cos(2π·hour/24) - cyclical time encoding
        [9] delta_htg        : Heating load demand (htg_sp - zone_avg)
        [10] delta_clg       : Cooling load demand (zone_avg - clg_sp)
        [11] delta_buf       : Buffer tank load demand (hw_T - buffer_T)
        [12] delta_dhw       : DHW tank load demand (dhw_sp - dhw_T)

    Action features (4):
        [13] htg_sp   : heating setpoint (°C)
        [14] clg_sp   : cooling setpoint (°C)
        [15] hw_T     : hot water supply temperature (°C)
        [16] dhw_sp   : DHW tank setpoint (°C)

    === OUTPUT PREDICTIONS (8 values) ===
        [0-3] next zone temperatures (living, kitchen, bedroom, bathroom)
        [4]   next buffer tank temperature
        [5]   next DHW tank temperature
        [6]   HVAC electric power (W)
        [7]   total facility electric power (W)
    """

    STATE_DIM = 13  # Güncellendi: 9'dan 13'e çıktı (4 yeni Delta T özelliği eklendi)
    ACTION_DIM = 4
    OUTPUT_DIM = 8
    OUTPUT_NAMES = (
        'zone_temp_living',
        'zone_temp_kitchen',
        'zone_temp_bedroom',
        'zone_temp_bathroom',
        'buffer_T',
        'dhw_T',
        'hvac_power',
        'total_power',
    )

    # Güncellendi: hidden_dim 128'den 256'ya çıkarıldı
    def __init__(self, ensemble_size=5, hidden_dim=256, learning_rate=5e-4,
                 output_weights=None):
        self.ensemble_size = ensemble_size
        self.input_dim = self.STATE_DIM + self.ACTION_DIM

        self.models = nn.ModuleList([
            ProbabilisticMLP(self.input_dim, self.OUTPUT_DIM, hidden_dim)
            for _ in range(ensemble_size)
        ])
        self.optimizer = torch.optim.Adam(
            self.models.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self._trained = False
        self.output_weights = self._build_output_weights(output_weights)
        self.last_split_data = None
        self.training_history = []
        self.best_val_loss = None

    def _build_output_weights(self, output_weights):
        if output_weights is None:
            output_weights = np.ones(self.OUTPUT_DIM, dtype=np.float32)
            output_weights[6] = 2.0
            output_weights[7] = 3.0
        weights = np.asarray(output_weights, dtype=np.float32)
        if weights.shape != (self.OUTPUT_DIM,):
            raise ValueError(
                'output_weights must have shape ({},), got {}'.format(
                    self.OUTPUT_DIM, weights.shape))
        return weights / weights.mean()

    @staticmethod
    def build_features(outdoor_T, zone_temps, buffer_T, dhw_T, hour,
                       htg_sp, clg_sp, hw_T, dhw_sp):
        """Construct the 17-dimensional input feature vector."""
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        # --- EKLENEN MÜHENDİSLİK ÖZELLİKLERİ (FEATURE ENGINEERING) ---
        avg_zone = np.mean(zone_temps)
        # Isıtma ve soğutma ihtiyacı (Negatif değerleri 0'a kırparak sadece aktif yükü gösteriyoruz)
        delta_htg = np.maximum(0.0, htg_sp - avg_zone) 
        delta_clg = np.maximum(0.0, avg_zone - clg_sp) 
        # Kazanların çalışma ihtiyacı
        delta_buf = np.maximum(0.0, hw_T - buffer_T)   
        delta_dhw = np.maximum(0.0, dhw_sp - dhw_T) 
        
        return np.array([
            outdoor_T,
            zone_temps[0], zone_temps[1], zone_temps[2], zone_temps[3],
            buffer_T, dhw_T,
            hour_sin, hour_cos,
            # Eklenen 4 yeni özellik
            delta_htg, delta_clg, delta_buf, delta_dhw,
            # Aksiyonlar
            htg_sp, clg_sp, hw_T, dhw_sp,
        ], dtype=np.float32)

    @staticmethod
    def build_targets(next_zone_temps, next_buffer_T, next_dhw_T,
                      hvac_power_W, total_power_W):
        """Construct the 8-dimensional output target vector."""
        return np.array([
            next_zone_temps[0], next_zone_temps[1],
            next_zone_temps[2], next_zone_temps[3],
            next_buffer_T, next_dhw_T,
            hvac_power_W, total_power_W,
        ], dtype=np.float32)

    def _set_statistics(self, inputs, outputs):
        """Compute and store normalization statistics.

        Z-score normalization: x_norm = (x - μ) / σ
        Applied to both inputs and outputs for stable training.
        The +1e-8 prevents division by zero for constant features.
        """
        self.input_mean = inputs.mean(axis=0).astype(np.float32)
        self.input_std = (inputs.std(axis=0) + 1e-8).astype(np.float32)
        self.output_mean = outputs.mean(axis=0).astype(np.float32)
        self.output_std = (outputs.std(axis=0) + 1e-8).astype(np.float32)

    def fit(self, inputs, outputs, epochs=150, batch_size=256, verbose=True,
            val_size=0.15, test_size=0.15, random_state=42):
        """Train all ensemble members with train/val/test splits and fixed bootstrap."""

        if val_size + test_size >= 1.0:
            raise ValueError('val_size + test_size must be < 1.0')

        inputs_tr, inputs_hold, outputs_tr, outputs_hold = train_test_split(
            inputs, outputs, test_size=val_size + test_size,
            random_state=random_state)

        if test_size > 0.0:
            relative_test_size = test_size / (val_size + test_size)
            inputs_val, inputs_test, outputs_val, outputs_test = train_test_split(
                inputs_hold, outputs_hold, test_size=relative_test_size,
                random_state=random_state)
        else:
            inputs_val, outputs_val = inputs_hold, outputs_hold
            inputs_test = np.empty((0, inputs.shape[1]), dtype=inputs.dtype)
            outputs_test = np.empty((0, outputs.shape[1]), dtype=outputs.dtype)

        # İstatistikleri SADECE eğitim verisi üzerinden hesapla
        self._set_statistics(inputs_tr, outputs_tr)
        self.last_split_data = {
            'train': (inputs_tr, outputs_tr),
            'val': (inputs_val, outputs_val),
            'test': (inputs_test, outputs_test),
        }
        self.training_history = []

        # Verileri yeni istatistiklere göre normalize et
        X_tr = (inputs_tr - self.input_mean) / self.input_std
        Y_tr = (outputs_tr - self.output_mean) / self.output_std
        X_val = (inputs_val - self.input_mean) / self.input_std
        Y_val = (outputs_val - self.output_mean) / self.output_std

        # PyTorch Tensörlerine çevir
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

        weight_t = torch.tensor(self.output_weights, dtype=torch.float32)

        # 2. DÜZELTME: Bootstrap indekslerini epoch döngüsü başlamadan önce belirle
        bootstrap_indices = []
        n_samples = len(X_tr_t)
        for _ in range(self.ensemble_size):
            # Her model için n_samples kadar veriyi yerine koyarak seç
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_indices.append(idx)

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            self.models.train()
            epoch_loss = 0.0

            # Her model için kendi sabit eğitim setini kullan
            for i, model in enumerate(self.models):
                
                # Bu modele özel önceden belirlenmiş verileri getir
                model_idx = bootstrap_indices[i].copy()
                
                # Veriyi her epoch'ta karıştırarak batch'lere böl (Eğitim kalitesini artırır)
                np.random.shuffle(model_idx) 

                for start in range(0, len(model_idx), batch_size):
                    end = min(start + batch_size, len(model_idx))
                    batch_idx = model_idx[start:end]
                    
                    xb = X_tr_t[batch_idx]
                    yb = Y_tr_t[batch_idx]

                    mean, logvar = model(xb)

                    # Gaussian NLL Loss
                    inv_var = torch.exp(-logvar)
                    mse_term = ((mean - yb) ** 2) * inv_var
                    weighted_nll = (mse_term + logvar) * weight_t
                    loss = weighted_nll.mean()

                    # Logvar sınırlarını düzenlileştirme (Regularization)
                    loss += 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()

            # --- Doğrulama (Validation) Aşaması ---
            self.models.eval()
            with torch.no_grad():
                val_losses = []
                for model in self.models:
                    mean, logvar = model(X_val_t)
                    inv_var = torch.exp(-logvar)
                    weighted_val = (((mean - Y_val_t)**2 * inv_var) + logvar) * weight_t
                    val_loss = weighted_val.mean()
                    val_losses.append(val_loss.item())
                avg_val_loss = np.mean(val_losses)
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(epoch_loss / self.ensemble_size),
                'val_loss': float(avg_val_loss),
            })

            # Early Stopping (Erken Durdurma) Kontrolü
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                best_state = {k: v.clone() for k, v in self.models.state_dict().items()}
            else:
                patience += 1

            if verbose and (epoch + 1) % 25 == 0:
                print('  Epoch {}/{} - train loss: {:.4f}, val loss: {:.4f}'.format(
                    epoch + 1, epochs,
                    epoch_loss / self.ensemble_size, avg_val_loss))

            if patience >= 20:
                if verbose:
                    print('  Early stopping at epoch {}'.format(epoch + 1))
                break

        self.models.load_state_dict(best_state)
        self._trained = True
        self.best_val_loss = best_val_loss
        if verbose:
            print('  Training complete. Best val loss: {:.4f}'.format(best_val_loss))

    def evaluate(self, inputs, outputs, split_name='dataset'):
        """Compute regression, residual, and uncertainty coverage metrics."""
        if len(inputs) == 0:
            return {
                'split': split_name,
                'num_samples': 0,
                'outputs': {},
            }

        preds, unc = self.predict(inputs)
        pred_matrix = np.column_stack([
            preds['next_zone_temps'],
            preds['next_buffer_T'],
            preds['next_dhw_T'],
            preds['hvac_power'],
            preds['total_power'],
        ])
        std_matrix = unc['total_std']

        metrics = {}
        for idx, name in enumerate(self.OUTPUT_NAMES):
            y_true = outputs[:, idx]
            y_pred = pred_matrix[:, idx]
            residuals = y_pred - y_true
            sigma = std_matrix[:, idx]
            metrics[name] = {
                'r2': float(r2_score(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_p05': float(np.percentile(residuals, 5)),
                'residual_p95': float(np.percentile(residuals, 95)),
                'coverage_1sigma': float(np.mean(np.abs(residuals) <= sigma)),
                'coverage_2sigma': float(np.mean(np.abs(residuals) <= 2.0 * sigma)),
            }

        aggregate = {
            'mean_r2': float(np.mean([m['r2'] for m in metrics.values()])),
            'mean_mae': float(np.mean([m['mae'] for m in metrics.values()])),
            'mean_rmse': float(np.mean([m['rmse'] for m in metrics.values()])),
        }

        return {
            'split': split_name,
            'num_samples': int(len(inputs)),
            'aggregate': aggregate,
            'outputs': metrics,
        }

    def evaluate_splits(self):
        """Evaluate the last train/val/test split captured during fit()."""
        if self.last_split_data is None:
            raise RuntimeError('No split data available. Call fit() first.')

        results = {}
        for split_name, (inputs, outputs) in self.last_split_data.items():
            results[split_name] = self.evaluate(inputs, outputs, split_name=split_name)
        return results

    @torch.no_grad()
    def predict(self, inputs):
        """Predict with uncertainty estimation.

        === PREDICTION MATH ===

        For input x, each ensemble member i produces:
            (μᵢ, σ²ᵢ) = f_θᵢ(x)

        Ensemble mean (best estimate):
            μ_ensemble = (1/N) Σᵢ μᵢ

        Total predictive variance (law of total variance):
            σ²_total = (1/N) Σᵢ σ²ᵢ        (mean aleatoric)
                     + (1/N) Σᵢ (μᵢ - μ̄)²  (epistemic)

        Returns:
            predictions: dict with mean values
            uncertainty: (N, 8) std deviations for each output
        """
        single = inputs.ndim == 1
        if single:
            inputs = inputs[np.newaxis, :]

        X = (inputs - self.input_mean) / self.input_std
        X_t = torch.tensor(X, dtype=torch.float32)

        self.models.eval()
        all_means = []
        all_vars = []

        for model in self.models:
            mean_norm, logvar_norm = model(X_t)
            # Unnormalize predictions
            mean = mean_norm.numpy() * self.output_std + self.output_mean
            var = np.exp(logvar_norm.numpy()) * (self.output_std ** 2)
            all_means.append(mean)
            all_vars.append(var)

        means = np.stack(all_means, axis=0)      # (E, N, 8)
        vars_arr = np.stack(all_vars, axis=0)     # (E, N, 8)

        # Ensemble mean
        ensemble_mean = means.mean(axis=0)        # (N, 8)

        # Total variance = mean(aleatoric) + var(epistemic)
        aleatoric = vars_arr.mean(axis=0)
        epistemic = means.var(axis=0)
        total_var = aleatoric + epistemic
        total_std = np.sqrt(total_var)

        predictions = {
            'next_zone_temps': ensemble_mean[:, 0:4],
            'next_buffer_T': ensemble_mean[:, 4],
            'next_dhw_T': ensemble_mean[:, 5],
            'hvac_power': ensemble_mean[:, 6],
            'total_power': ensemble_mean[:, 7],
        }

        uncertainty = {
            'total_std': total_std,
            'epistemic_std': np.sqrt(epistemic),
            'aleatoric_std': np.sqrt(aleatoric),
            'zone_temp_std': total_std[:, 0:4],
            'power_std': total_std[:, 7],
        }

        return predictions, uncertainty

    def predict_power(self, outdoor_T, zone_temps, buffer_T, dhw_T, hour,
                      htg_sp, clg_sp, hw_T, dhw_sp):
        """Convenience: predict for a single condition with uncertainty."""
        feat = self.build_features(
            outdoor_T, zone_temps, buffer_T, dhw_T, hour,
            htg_sp, clg_sp, hw_T, dhw_sp)
        preds, unc = self.predict(feat)
        return {
            'next_zone_temps': preds['next_zone_temps'][0],
            'next_buffer_T': float(preds['next_buffer_T'][0]),
            'next_dhw_T': float(preds['next_dhw_T'][0]),
            'hvac_power': float(preds['hvac_power'][0]),
            'total_power': float(preds['total_power'][0]),
            'power_std': float(unc['power_std'][0]),
            'zone_temp_std': unc['zone_temp_std'][0],
        }

    def save(self, path):
        torch.save({
            'models': self.models.state_dict(),
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'ensemble_size': self.ensemble_size,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.models.load_state_dict(ckpt['models'])
        self.input_mean = ckpt['input_mean']
        self.input_std = ckpt['input_std']
        self.output_mean = ckpt['output_mean']
        self.output_std = ckpt['output_std']
        self._trained = True
