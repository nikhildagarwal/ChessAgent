import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8):
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
        self.layer3 = nn.Linear(390, 200)
        self.layer4 = nn.Linear(200, 150)
        self.layer5 = nn.Linear(150, 100)
        self.layer6 = nn.Linear(100, 64)

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, 65) or (batch, 65).
           The first 64 elements are the main features for attention and the 65th element is a binary flag.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        main_features = x[..., :64]
        binary_features = x[..., 64]
        extra_feature = binary_features[:, -1].unsqueeze(1)
        attn_output, attn_weights = self.attention(query=main_features, key=main_features, value=main_features)
        context = attn_output.mean(dim=1)
        context = torch.cat([context, extra_feature], dim=1)
        x = torch.sigmoid(self.layer1(context))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))
        x = self.layer6(x)
        x = F.log_softmax(x, dim=1)
        return x, attn_weights

    def save_model(self, filepath):
        """
        Saves the model's state dictionary to the given filepath.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, embed_dim=64, num_heads=8, to_train=False):
        """
        Loads the model's state dictionary from the given filepath and returns an instance.
        """
        model = cls(embed_dim=embed_dim, num_heads=num_heads)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        if to_train:
            model.train()
        return model
