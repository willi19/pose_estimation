import torch

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        pred_p = pred.permute(1,0,2,3,4)
        l = ((pred_p - gt)**2)
        l = l.mean(dim=0).mean(dim=1).mean(dim=1).mean(dim=1)
        return l 

def get_max(hmap):
    val, x = hmap.max(2)
    y = torch.unsqueeze(val.max(2)[1],0)
    x = torch.gather(x,2,y)
    return torch.stack([x,y],dim=1)


def get_accuracy(pred, gt):
    final_pred = pred[:,-1]        #batch 24 img_size
    pred_cor = get_max(final_pred) 
    gt_cor = get_max(gt)
    return ((pred_cor-gt_cor)**2).sum().float()