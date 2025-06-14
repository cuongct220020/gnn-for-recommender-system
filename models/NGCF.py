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

class NGCFConv(MessagePassing):
    def __init__(self,
                 embedding_dim: int = 64,
                 aggr: str = 'add', flow: str = 'source_to_target',
                 mess_dropout_prob: float = 0.2,
                 node_dropout_prob: float = 0.6,
                 norm = gcn_norm):
        super(NGCFConv, self).__init__(aggr=aggr, flow=flow)
        self.embedding_dim = embedding_dim
        self.mess_dropout_prob = mess_dropout_prob
        self.node_dropout_prob = node_dropout_prob
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.norm = norm
        # Weight matrices W1 and W2
        self.lin1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Message Dropout Layer
        self.mess_dropout_layer = nn.Dropout(self.mess_dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x: Tensor,
                edge_index: SparseTensor,
                edge_weight: Tensor = None) -> Tensor:
        if isinstance(edge_index, SparseTensor):
            # === Apply Normalization ===
            edge_index = self.norm(edge_index, None, x.size(self.node_dim),
                                    add_self_loops=False, flow=self.flow, dtype=x.dtype)

            # === Node Dropout ===
            if self.training and self.node_dropout_prob > 0:
                # Get edge indices from SparseTensor
                row, col, _ = edge_index.coo()
                ei = torch.stack([row, col], dim=0)

                # Apply node dropout
                ei, edge_mask, node_mask = dropout_node(
                    ei,
                    p=self.node_dropout_prob,
                    num_nodes=x.size(self.node_dim),
                    training=True
                )

                # Create new SparseTensor with dropped edges
                if edge_weight is not None:
                    edge_weight = edge_weight[edge_mask] if edge_weight.size(0) == edge_mask.size(0) else edge_weight

                edge_index = SparseTensor.from_edge_index(
                    edge_index=ei,
                    sparse_sizes=edge_index.sizes()
                ).to(ei.device)
        else:
            raise ValueError("Unsupported edge_index type.")

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor,
                x_i: Tensor, edge_weight: Tensor = None) -> Tensor:
        msg = self.lin1(x_j) + self.lin2(x_i * x_j)
        if edge_weight is not None:
            msg = edge_weight.view(-1, 1) * msg
        # === Message Dropout ===
        if self.training and self.mess_dropout_prob > 0:
            msg = self.mess_dropout_layer(msg)
        return msg

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        msg = self.lin1(x) + self.lin2(x * x)
        # === Message Dropout ===
        if self.training and self.mess_dropout_prob > 0:
            msg = self.mess_dropout_layer(msg)
        return spmm(adj_t, msg, reduce=self.aggr)

    def update(self, aggr_out: Tensor) -> Tensor:
        return self.activation(aggr_out)

class NGCF(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 64,
                 num_layers: int = 3,
                 mess_dropout_prob: float = 0.2,
                 node_dropout_prob: float = 0.4,
                 norm=gcn_norm):
        super(NGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        # Embedding Layer
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        # Embedding Propagation Layer
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                NGCFConv(embedding_dim=embedding_dim,
                         mess_dropout_prob=mess_dropout_prob,
                         node_dropout_prob=node_dropout_prob,
                         norm=norm)
            )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index: SparseTensor, edge_weight: Tensor = None) -> Tensor:
        """
        Thực hiện L bước propagation và trả về tensor [N, (L+1)*D]
        chứa embedding gốc (layer 0) và embedding sau mỗi convolution.
        """
        x = self.embedding.weight
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_emb.append(x)
        return torch.cat(all_emb, dim=1)

    def forward(self, edge_index: SparseTensor,
                edge_label_index: Tensor,
                edge_weight: Tensor = None) -> Tensor:
        emb = self.get_embedding(edge_index, edge_weight)
        emb_src = emb[edge_label_index[0]]
        emb_dist = emb[edge_label_index[1]]
        scores = (emb_src * emb_dist).sum(dim=-1)
        return scores

    def compute_bpr_loss(self, pos_scores: Tensor, neg_scores: Tensor, lambda_reg: float = 1e-3) -> Tensor:
        """
        Tính BPR Loss, bao gồm cả phần regularization L2 cho TẤT CẢ các tham số có thể học được của mô hình.
        """
        loss_fn = BPRLoss(lambda_reg=lambda_reg)

        # Thu thập TẤT CẢ các tham số của mô hình
        all_learnable_parameters = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                # print(f"Regularizing parameter: {name}, shape: {param.shape}")
                all_learnable_parameters.append(param.view(-1)) # Làm phẳng và thêm vào danh sách

        # Nối tất cả các tensor tham số đã làm phẳng thành một tensor lớn
        if all_learnable_parameters:
            concatenated_params = torch.cat(all_learnable_parameters)
        else:
            concatenated_params = None # Không có tham số để regularize

        return loss_fn(pos_scores, neg_scores, concatenated_params)
