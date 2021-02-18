import torch

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, pred, gt):
        pred_m = pred.mean(1)
        l = self.loss(pred_m, gt)
        return l 

def get_max_cor(hmap):
    val, x = hmap.max(2)
    y = torch.unsqueeze(val.max(2)[1],0)
    x = torch.gather(x,2,y)
    return torch.stack([x,y],dim=1).float()


def get_accuracy(pred, gt):
    final_pred = pred[:,-1]        #batch 24 img_size
    pred_cor = get_max_cor(final_pred) 
    gt_cor = get_max_cor(gt)
    loss = torch.nn.MSELoss()
    return loss(pred_cor, gt_cor)