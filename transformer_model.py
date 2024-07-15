import numpy as np
# Bring in PyTorch
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, embeddings_prot_bert: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        Q = torch.matmul(embeddings_prot_bert, self.w_q)
        K = torch.matmul(embeddings_prot_bert, self.w_k)
        V = torch.matmul(embeddings_prot_bert, self.w_v)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1))
        scores = scores.masked_fill(mask == 0, float('-inf')) #if masked, the value is going as low as possible to avoid being scored

        attn = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attn, V)

        return attention_output



class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self,idx, predicted_angles, angles_tensor, mask):

        predicted_angles_phi, predicted_angles_psi = predicted_angles[:, 0], predicted_angles[:, 1]
        angles_tensor_phi, angles_tensor_psi = angles_tensor[:,0, idx], angles_tensor[:, 1,idx]
        
        predicted_angles_phi = (predicted_angles_phi + torch.pi) % (2 * torch.pi) - torch.pi
        angles_tensor_phi = (angles_tensor_phi + torch.pi) % (2 * torch.pi) - torch.pi
        predicted_angles_psi = (predicted_angles_psi + torch.pi) % (2 * torch.pi) - torch.pi
        angles_tensor_psi = (angles_tensor_psi + torch.pi) % (2 * torch.pi) - torch.pi

        difference_phi = torch.abs(predicted_angles_phi - angles_tensor_phi)*mask
        loss_phi = torch.mean(torch.min(difference_phi, 2 * torch.pi - difference_phi)) / mask.sum()

        difference_psi = torch.abs(predicted_angles_psi - angles_tensor_psi)*mask
        loss_psi = torch.mean(torch.min(difference_psi, 2 * torch.pi - difference_psi))/ mask.sum()

        loss = loss_phi + loss_psi
        #print(loss)
        return loss
    
class TransformerModel(nn.Module):
    def __init__(self, embed_dim: int, feed_forward_dim1: int, feed_forward_dim2: int, output_dim: int = 2, dropout_rate: float = 0.1):
        super(TransformerModel, self).__init__()
        self.self_attention = SelfAttention(embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.feed_forward = nn.Sequential(  
            nn.Linear(embed_dim, feed_forward_dim1),
            nn.GELU(),
            self.dropout,
            nn.Linear(feed_forward_dim1, feed_forward_dim2),
            nn.GELU(),
            self.dropout,
            nn.Linear(feed_forward_dim2, output_dim)
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attention(embeddings, mask)
        normalized_attention_output = self.layer_norm1(attention_output)
        ff_output = self.feed_forward(normalized_attention_output)
        #output = self.layer_norm2(ff_output)
        return ff_output
    