import numpy as np
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    '''
    Input =>
    embed_dim : the dimension of the embedding space (1024)
    embeddings : sequence embeddings
    mask : masks for the given sequence

    Output =>
    attention_output : attention result
    '''
    def __init__(self, embed_dim: int):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        #Initializing the query, key and value weight parameters
        self.w_q = nn.Parameter(torch.randn(embed_dim, embed_dim)) 
        self.w_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.w_v = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        #Calculating the query, key and value
        Q = torch.matmul(embeddings, self.w_q)
        K = torch.matmul(embeddings, self.w_k)
        V = torch.matmul(embeddings, self.w_v)
        #Creating the attention mask
       

        #Attention score calculation
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        scores = scores + mask.T
        #scores = scores.masked_fill(mask == 0, float('-inf'))
        
        #Attention calculation
        attn = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attn, V)
        
        return attention_output


class AngularLoss(nn.Module):
    '''
    Input =>
    idx : index of the sequence in the seqeunce list
    predicted_angles : predicted angles
    angles_tensor : original angles (in radians)
    mask : mask to ignore the added paddings

    Output =>
    loss : loss calculated for phi and psi angles
    '''
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self,idx, predicted_angles, angles_tensor, mask):

        #Separating phi and psi angles
        predicted_angles_phi, predicted_angles_psi = predicted_angles[:, 0], predicted_angles[:, 1]
        angles_tensor_phi, angles_tensor_psi = angles_tensor[:,0, idx], angles_tensor[:, 1,idx]
        
        #Putting the angles in the thorus
        predicted_angles_phi = (predicted_angles_phi + torch.pi) % (2 * torch.pi) - torch.pi
        angles_tensor_phi = (angles_tensor_phi + torch.pi) % (2 * torch.pi) - torch.pi
        predicted_angles_psi = (predicted_angles_psi + torch.pi) % (2 * torch.pi) - torch.pi
        angles_tensor_psi = (angles_tensor_psi + torch.pi) % (2 * torch.pi) - torch.pi

        #Finding the thorus distance of prediction and angle values
        difference_phi = torch.abs(predicted_angles_phi - angles_tensor_phi)*mask
        loss_phi = torch.sum(torch.min(difference_phi, 2 * torch.pi - difference_phi)) / mask.sum()

        difference_psi = torch.abs(predicted_angles_psi - angles_tensor_psi)*mask
        loss_psi = torch.sum(torch.min(difference_psi, 2 * torch.pi - difference_psi))/ mask.sum()

        loss = loss_phi + loss_psi
        return loss
    
class TransformerModel(nn.Module):
    '''
    Input =>
    embed_dim : the dimension of the embedding space (1024)
    feed_forward_dim1 : size of the first layer of FFNN
    feed_forward_dim2 : size of the hidden layer of FFNN
    output_dim : 2, since we have 2 angles for each residue
    dropout_dim : MEANING

    Output =>
    ff_output : transformed embedding with size [0,129,2]
    '''
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
    