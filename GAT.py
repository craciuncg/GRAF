import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    
    def __init__(
        self,
        in_features,
        out_features,
        n_heads: int,
        concat: bool = False,
        dropout: float = 0.4,
        leaky_relu_slope: float = 0.2,
        device='cpu',
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate
        self.leaky_relu_slope = leaky_relu_slope
        
        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.WN = nn.Parameter(torch.empty(size=(self.in_features, self.n_hidden * n_heads))).to(device)
        self.WE = nn.Parameter(torch.empty(size=(self.in_features, self.n_hidden * n_heads))).to(device)
        
        self.aN = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1))).to(device)
        self.aE = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1))).to(device)
        
        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
        self.softmax = nn.Softmax(dim=1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.WE)
        nn.init.xavier_normal_(self.aE)
        
        nn.init.xavier_normal_(self.WN)
        nn.init.xavier_normal_(self.aN)
    
    def _get_attention_scores(
        self,
        h_transformed1: torch.Tensor,
        h_transformed2: torch.Tensor,
        a: torch.Tensor
    ):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed1, a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed2, a[:, self.n_hidden:, :])

        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)
    
    def forward(
        self,
        nodes_embeddings,
        edges_embeddings,
        mat_nodes,
        mat_edges
    ):
        n_nodes = nodes_embeddings.shape[0]
        n_edges = edges_embeddings.shape[0]
        
        nodes_transformed = torch.mm(nodes_embeddings, self.WN)
        edges_transformed = torch.mm(edges_embeddings, self.WE)
        
        nodes_transformed = F.dropout(nodes_transformed, self.dropout, training=self.training)
        edges_transformed = F.dropout(edges_transformed, self.dropout, training=self.training)
        
        nodes_transformed = nodes_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        edges_transformed = edges_transformed.view(n_edges, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        e_N = self._get_attention_scores(nodes_transformed, nodes_transformed, self.aN)
        e_E = self._get_attention_scores(nodes_transformed, edges_transformed, self.aE)
        
        connectivity_mask_N = -9e16 * torch.ones_like(e_N)
        e_N = torch.where(mat_nodes > 0, e_N, connectivity_mask_N)
        
        connectivity_mask_E = -9e16 * torch.ones_like(e_E)
        e_E = torch.where(mat_edges > 0, e_E, connectivity_mask_E)
        
        attention_E = F.softmax(e_E, dim=-1)
        attention_E = F.dropout(attention_E, self.dropout, training=self.training)
        
        attention_N = F.softmax(e_N, dim=-1)
        attention_N = F.dropout(attention_N, self.dropout, training=self.training)
        
        h_prime_E = torch.matmul(attention_E, edges_transformed)
        h_prime_N = torch.matmul(attention_N, nodes_transformed)
        
        h_prime = h_prime_E + h_prime_N
        
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime