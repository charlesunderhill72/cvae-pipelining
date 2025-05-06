import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    A basic convolutional variational autoencoder. 
    Works best with inputs padded to heights and widths 
    that are a power of 2.
    """
    def __init__(self, image_channels=2, init_channels=32, kernel_size=3, latent_dim=128):
        super(ConvVAE, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(image_channels, init_channels, kernel_size=kernel_size, stride=2, padding=1)  # Output: [batch, 32, 64, 128]
        self.enc2 = nn.Conv2d(init_channels, init_channels * 2, kernel_size=kernel_size, stride=2, padding=1)  # Output: [batch, 64, 32, 64]
        self.enc3 = nn.Conv2d(init_channels * 2, init_channels * 4, kernel_size=kernel_size, stride=2, padding=1)  # Output: [batch, 128, 16, 32]
        self.enc4 = nn.Conv2d(init_channels * 4, 256, kernel_size=kernel_size, stride=2, padding=1)  # Output: [batch, 256, 8, 16]

        # Latent space
        self.fc_mu = nn.Linear(256 * 8 * 16, latent_dim)
        self.fc_log_var = nn.Linear(256 * 8 * 16, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 256 * 8 * 16)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, init_channels * 4, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)  # Output: [batch, 128, 16, 32]
        self.dec2 = nn.ConvTranspose2d(init_channels * 4, init_channels * 2, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)  # Output: [batch, 64, 32, 64]
        self.dec3 = nn.ConvTranspose2d(init_channels * 2, init_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)  # Output: [batch, 32, 64, 128]
        self.dec4 = nn.ConvTranspose2d(init_channels, image_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)  # Output: [batch, 2, 128, 256]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))

        # Flatten and pass through fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Decoder
        z = self.fc2(z)  # Reshape to match the size after encoding
        z = z.view(-1, 256, 8, 16)  # Reshape to [batch_size, 256, 8, 16]

        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))

        return reconstruction, mu, log_var

