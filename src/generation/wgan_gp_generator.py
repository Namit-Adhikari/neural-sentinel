import os
import pandas as pd
import numpy as np
import logging
from .base_generator import BaseGenerator

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class WGANGP_Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)

class WGANGP_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class WGANGPGenerator(BaseGenerator):
    """
    Custom WGAN-GP implementation for tabular data using PyTorch.
    Supports basic numeric and categorical mixed types via scikit-learn transformers.
    """
    def __init__(self, epochs=30, batch_size=256, latent_dim=128, lr=1e-4, n_critic=5, lambda_gp=10):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and scikit-learn are required for WGAN-GP. Please install torch and scikit-learn.")
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        
        self.generator = None
        self.discriminator = None
        self.preprocessor = None
        self.is_fitted = False
        self.columns = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones((real_samples.size(0), 1)).to(self.device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def fit(self, data: pd.DataFrame):
        self.columns = data.columns.tolist()
        
        # Prepare preprocessor
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        encoded_data = self.preprocessor.fit_transform(data)
        encoded_data = encoded_data.astype(np.float32)
        output_dim = encoded_data.shape[1]
        
        dataset = TensorDataset(torch.tensor(encoded_data))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self.generator = WGANGP_Generator(self.latent_dim, output_dim).to(self.device)
        self.discriminator = WGANGP_Discriminator(output_dim).to(self.device)
        
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        
        logging.info("Starting WGAN-GP training...")
        for epoch in range(self.epochs):
            for i, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                current_batch_size = real_batch.size(0)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_batch = self.generator(z).detach()
                
                real_validity = self.discriminator(real_batch)
                fake_validity = self.discriminator(fake_batch)
                
                gp = self._compute_gradient_penalty(real_batch, fake_batch)
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gp
                d_loss.backward()
                optimizer_D.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                if i % self.n_critic == 0:
                    optimizer_G.zero_grad()
                    z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                    fake_batch = self.generator(z)
                    
                    fake_validity = self.discriminator(fake_batch)
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    optimizer_G.step()
                    
        self.is_fitted = True
        
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
            
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_rows, self.latent_dim).to(self.device)
            fake_data = self.generator(z).cpu().numpy()
            
        # Inverse transform
        decoded_data = self.preprocessor.inverse_transform(fake_data)
        df = pd.DataFrame(decoded_data, columns=self.columns)
        
        # Cast numeric cols back
        for col in df.columns:
            if col in self.preprocessor.transformers_[0][2]:  # numeric columns
                df[col] = pd.to_numeric(df[col])
                
        return df
        
    def save(self, path: str):
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = {
            'generator': self.generator.state_dict(),
            'preprocessor': self.preprocessor,
            'columns': self.columns,
            'latent_dim': self.latent_dim
        }
        torch.save(model_state, path)
        
    @classmethod
    def load(cls, path: str) -> "WGANGPGenerator":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")
        instance = cls()
        model_state = torch.load(path)
        
        instance.preprocessor = model_state['preprocessor']
        instance.columns = model_state['columns']
        instance.latent_dim = model_state['latent_dim']
        
        output_dim = len(instance.preprocessor.get_feature_names_out())
        instance.generator = WGANGP_Generator(instance.latent_dim, output_dim).to(instance.device)
        instance.generator.load_state_dict(model_state['generator'])
        instance.generator.eval()
        instance.is_fitted = True
        return instance
