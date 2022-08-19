import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity()

    def forward(self, anchor, pos, neg):
        pos_loss = 1.0 - self.cos(anchor, pos).mean(dim=0)
        neg_loss = self.cos(anchor, neg).mean(dim=0)
        return pos_loss + neg_loss