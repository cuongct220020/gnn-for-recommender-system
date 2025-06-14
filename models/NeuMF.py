import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers, dropout):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        # GMF embeddings
        self.user_emb_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.user_emb_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.item_emb_mlp = nn.Embedding(num_items, layers[0] // 2)

        # MLP layers
        mlp = []
        for i in range(len(layers) - 1):
            mlp.append(nn.Linear(layers[i], layers[i+1]))
            mlp.append(nn.ReLU())
            if dropout:
                mlp.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp)

        # Prediction layer
        self.head = nn.Linear(layers[-1] + embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.user_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.item_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

        # Initialize MLP layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Initialize prediction layer
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, user_idx, item_idx):
        # GMF path
        user_emb_gmf = self.user_emb_gmf(user_idx)  # [batch_size, embedding_dim]
        item_emb_gmf = self.item_emb_gmf(item_idx)  # [batch_size, embedding_dim]
        gmf_vector = user_emb_gmf * item_emb_gmf    # [batch_size, embedding_dim]

        # MLP path
        user_emb_mlp = self.user_emb_mlp(user_idx)  # [batch_size, layers[0]//2]
        item_emb_mlp = self.item_emb_mlp(item_idx)  # [batch_size, layers[0]//2]
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)  # [batch_size, layers[0]]
        mlp_vector = self.mlp(mlp_vector)  # [batch_size, layers[-1]]

        # Combine paths
        vector = torch.cat([gmf_vector, mlp_vector], dim=1)  # [batch_size, layers[-1] + embedding_dim]
        scores = self.head(vector).squeeze(-1)  # [batch_size]
        return scores

    def compute_bpr_loss(self, pos_scores: torch.Tensor,
                         neg_scores: torch.Tensor,
                         lambda_reg: float = 1e-3) -> torch.Tensor:
        loss_fn = BPRLoss(lambda_reg=lambda_reg)

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
                pos_score: Tensor,
                neg_score: Tensor,
                parameters: Tensor = None) -> Tensor:
        """
        pos_score: [batch_size]
        neg_score: [batch_size, num_neg]
        parameters: embedding cần regularize
        """
        # Ensure pos_score has shape [batch_size, 1] for broadcasting
        if pos_score.dim() == 1:
            pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]

        log_prob = F.logsigmoid(pos_score - neg_score).mean()

        regularization = 0
        if self.lambda_reg != 0 and parameters is not None:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)
            regularization = regularization / pos_score.size(0)

        return -log_prob + regularization