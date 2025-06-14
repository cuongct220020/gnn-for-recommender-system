import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Tensor, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class BPRLoss(_Loss):
    def __init__(self, lambda_reg: float = 1e-3):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self,
                pos_score: torch.Tensor,
                neg_score: torch.Tensor,
                parameters: torch.Tensor = None) -> torch.Tensor:
        if pos_score.dim() == 1:
            pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]

        log_prob = F.logsigmoid(pos_score - neg_score).mean()

        regularization = 0
        if self.lambda_reg != 0 and parameters is not None:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / pos_score.size(0)

        return -log_prob + regularization


class GCMC(MessagePassing):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 64,
                 aggr: str = 'add',
                 flow: str = 'source_to_target',
                 norm = gcn_norm,
                 message_dropout: float = 0.1,
                 node_dropout: float = 0.1):
        super(GCMC, self).__init__(aggr=aggr, flow=flow)

        # Model parameters
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.norm = norm

        # Dropout layers
        self.message_dropout = nn.Dropout(message_dropout)
        self.node_dropout = nn.Dropout(node_dropout)

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # Node embeddings
        self.embedding = nn.Embedding(self.num_nodes, self.embedding_dim)

        # Encoder components
        self.W_pos = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_dense = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all model parameters with proper scaling"""
        # Use smaller initialization for better gradient flow
        nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_pos.weight, gain=0.1)
        nn.init.xavier_uniform_(self.W_dense.weight, gain=0.1)

    def encode(self, x: Tensor, edge_index: SparseTensor, edge_weight: Tensor = None) -> Tensor:
        """Encoder forward pass with improved normalization"""
        # Apply node dropout to input embeddings
        x = self.node_dropout(x)

        if isinstance(edge_index, SparseTensor):
            edge_index = self.norm(edge_index, None, x.size(self.node_dim),
                                 add_self_loops=False, flow=self.flow, dtype=x.dtype)
        else:
            raise ValueError("Unsupported edge_index type.")

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor) -> Tensor:
        """Message function with dropout"""
        msg = self.W_pos(x_j)
        return self.message_dropout(msg)

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        """Combined message passing and aggregation with dropout"""
        msg = self.W_pos(x)
        msg = self.message_dropout(msg)
        return spmm(adj_t, msg, reduce=self.aggr)

    def update(self, aggr_out: Tensor) -> Tensor:
        """Update function with residual connection"""
        # Apply transformation
        transformed = self.W_dense(aggr_out)
        # Apply activation
        output = self.activation(transformed)
        return output

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        """Decoder with L2 normalization for better stability"""
        users_emb = z[edge_label_index[0]]
        items_emb = z[edge_label_index[1]]

        # L2 normalize embeddings for better stability
        users_emb = F.normalize(users_emb, p=2, dim=-1)
        items_emb = F.normalize(items_emb, p=2, dim=-1)

        # Compute dot product scores
        scores = (users_emb * items_emb).sum(dim=-1)
        return scores

    def forward(self, adj: SparseTensor, edge_label_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        """Complete forward pass"""
        # Get node embeddings
        x = self.embedding.weight

        # Encode node features
        z = self.encode(x, adj, edge_weight)

        # Decode edge scores
        scores = self.decode(z, edge_label_index)

        return scores

    def compute_bpr_loss(self, pos_scores: Tensor, neg_scores: Tensor, lambda_reg: float = 1e-3) -> Tensor:
        loss_fn = BPRLoss(lambda_reg=lambda_reg)

        # Collect learnable parameters for regularization
        all_learnable_parameters = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_learnable_parameters.append(param.view(-1))

        # Concatenate all parameters
        if all_learnable_parameters:
            concatenated_params = torch.cat(all_learnable_parameters)
        else:
            concatenated_params = None

        return loss_fn(pos_scores, neg_scores, concatenated_params)