import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=64):
        """
        embed_dim: Dimensionality for the attention module (first 64 elements).
        num_heads: Number of attention heads (must divide embed_dim evenly).
        """
        super(ModelAttention, self).__init__()
        # Multi-head self-attention expects inputs of shape (batch, seq_len, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Fully connected layers.
        # layer1 now expects an input of dimension embed_dim + 1 = 65.
        self.layer1 = nn.Linear(embed_dim + 1, 130)
        self.layer2 = nn.Linear(130, 390)
        self.layer3 = nn.Linear(390, 600)
        self.layer4 = nn.Linear(600, 400)
        self.layer5 = nn.Linear(400, 200)
        self.layer6 = nn.Linear(200, 150)
        self.layer7 = nn.Linear(150, 100)
        self.layer8 = nn.Linear(100, 64)

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, 65) or (batch, 65).
           The first 64 elements are the main features for attention and the 65th element is a binary flag.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        main_features = x[..., :64]
        # Instead of extracting from binary_features, set extra_feature to a constant 0.5.
        extra_feature = torch.full((x.shape[0], 1), 0.5, device=x.device)
        attn_output, attn_weights = self.attention(query=main_features, key=main_features, value=main_features)
        context = attn_output.mean(dim=1)
        context = torch.cat([context, extra_feature], dim=1)
        x = torch.sigmoid(self.layer1(context))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = self.layer8(x)
        x = F.log_softmax(x, dim=1)
        return x, attn_weights

    def save_model(self, filepath):
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, embed_dim=64, num_heads=64, to_train=False):
        """
        Loads the model's state dictionary from the given filepath and returns an instance.
        """
        model = cls(embed_dim=embed_dim, num_heads=num_heads)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        if to_train:
            model.train()
        return model
