import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (with Logits).
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Default: 2.0 (from paper/screenshot)
        alpha (float): Class weight for positive class (Toxic).
                       1:1 balanced data → alpha=1.0 (equal weight)
                       Unbalanced data → alpha=positive/negative ratio
        reduction (str): 'mean' or 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch_size, 1) or (batch_size, )
        targets: (batch_size, 1) or (batch_size, ) - values in [0, 1]
        """
        # BCEWithLogitsLoss equivalent with raw sigmoid
        probs = torch.sigmoid(logits)
        targets = targets.view_as(logits)
        
        # p_t: probability of being the correct class
        # if y=1, p_t = p; if y=0, p_t = 1-p
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # alpha_t: weighting
        # if alpha is provided (e.g. 12.5), we apply it to Positive class (y=1)
        # and 1.0 to Negative class (y=0).
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * 1.0
        else:
            alpha_t = 1.0
            
        # fl = - alpha_t * (1 - p_t)^gamma * log(p_t)
        # Note: BCE loss term is -log(p_t). 
        # So FL = alpha_t * (1 - p_t)^gamma * BCE
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
