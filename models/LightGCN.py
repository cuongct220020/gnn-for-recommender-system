import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Optional, Union, Tensor, SparseTensor
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
    
class LightGCNConv(MessagePassing):
  def __init__(self,
               aggr: str = 'add',
               flow: str = 'source_to_target',
               norm = gcn_norm):
    super(LightGCNConv, self).__init__(aggr=aggr, flow=flow)
    self.aggr = aggr
    self.flow = flow
    self.norm = norm

  def forward(self,
            x: Tensor,
            edge_index: SparseTensor,
            edge_weight: Tensor = None) -> Tensor:
    """edge_index biểu diễn cấu trúc đồ thị (ma trận kề)"""
    if isinstance(edge_index, SparseTensor):
      edge_index = self.norm(edge_index, None, x.size(self.node_dim),
                            add_self_loops = False, flow=self.flow, dtype=x.dtype)
    else:
      raise ValueError("Unsupported edge_index type.")
    return self.propagate(edge_index, x=x, edge_weight = edge_weight)

  def message(self,
              x_j: Tensor,
              edge_weight: Tensor = None) -> Tensor:
    """x_j là vector đặc trưng của nút nguồn"""
    return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

  def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    return spmm(adj_t, x, reduce = self.aggr)
  
class LightGCN(nn.Module):
    def __init__(self,
                num_users: int,
                num_items: int,
                embedding_dim: int = 64,
                num_layers: int = 3,
                alpha: Optional[Union[float, Tensor]] = None):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers

        # Alpha weights cho layer-combination
        if alpha is None:
            alpha = 1 / (num_layers + 1)
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        # Embedding Layer
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        # Convolutional Layer
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self,
                    edge_index: SparseTensor,
                    edge_weight: Tensor = None) -> Tensor:
        """
        Triển khai Layer Combination, kết hợp embedding được học ở mỗi layer,
        tạo ra embedding cuối cùng cho mỗi nút.
        """
        x = self.embedding.weight # Lấy Embedding ban đầu (tầng 0)
        final_emb = x * self.alpha[0] # Embedding ban đầu nhân với alpha[0]

        for i, conv in enumerate(self.convs):  # Duyệt qua các lớp trong self.convs
            x = conv(x, edge_index, edge_weight) # Gọi đến LighGCNConv
            final_emb = final_emb + x * self.alpha[i + 1] # Cộng dồn embedding sau mỗi tầng
        return final_emb

    def forward(self,
                edge_index: Adj,
                edge_label_index: Tensor,
                edge_weight: Tensor) -> Tensor:
        """
        edge_index: Ma trận kề thưa dạng SparseTensor
        edge_label_index: cặp (u, i) muốn tính điểm dự đoán
        """
        emb = self.get_embedding(edge_index, edge_weight)
        emb_src = emb[edge_label_index[0]]
        emb_dist = emb[edge_label_index[1]]
        scores = (emb_src * emb_dist).sum(dim=-1)
        return scores

    def compute_bpr_loss(self, pos_scores: Tensor, neg_scores: Tensor,
                      node_id: Tensor = None, lambda_reg = 1e-3) -> Tensor:
        loss_fn = BPRLoss(lambda_reg=lambda_reg)
        emb = self.embedding.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_scores, neg_scores, emb)
    
