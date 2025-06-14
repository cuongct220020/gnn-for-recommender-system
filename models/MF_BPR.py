import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class MF_BPR(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MF_BPR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User embedding, Item embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]

        # Compute interaction
        scores = (user_emb * item_emb).sum(dim=1)  # [batch_size]
        return scores

    def compute_bpr_loss(self,
                         pos_scores: torch.Tensor,
                         neg_scores: torch.Tensor,
                         lambda_reg: float = 1e-3) -> torch.Tensor:
        loss_fn = BPRLoss(lambda_reg=lambda_reg)

        # Collect all learnable parameters
        all_params = []
        for param in self.parameters():
            if param.requires_grad:
                all_params.append(param.view(-1))

        concatenated_params = torch.cat(all_params) if all_params else None
        return loss_fn(pos_scores, neg_scores, concatenated_params)
    
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
