import torch
from data_helper import DataHelper
from dataset import MyDataset
from torch.utils.data import DataLoader
import utils
import metrics
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# you can load your model here
model = torch.load('reuters300d-ngram=2-T=2-direction=backward-dropout=0.5-AB=0.4-0.5-label_node=3-constrative=0stopword.pth')
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)


def collate_fn(batch):
    #  redefine collate_fn
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1])
    texts = list(batch[0])
    return texts, labels


model.cuda()
data_helper = DataHelper(dataset='reuters', mode='test')
test_dataset = MyDataset(data_helper)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=16,
                             pin_memory=False, collate_fn=collate_fn)
iter = 0
pred_all = torch.tensor([]).cuda()
label_all = torch.tensor([]).cuda()
model.eval()
with torch.no_grad():
    # for content, label, _ in data_helper.batch_iter(batch_size=128, num_epoch=1):
    for content, label in test_dataloader:
        iter += 1
        label = label.cuda()
        label_emp = []
        _, logits = model(content, label_emp, mode='dev/test')

        pred_all = torch.cat((pred_all, logits), dim=0)
        label_all = torch.cat((label_all, label), dim=0)


test_hl, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, test_micpre, test_macpre, \
    test_truenum_micpre, test_truenum_macpre, test_sampre, \
    test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
    test_macroauc, test_microauc, test_class = metrics.get_metrics(pred_all, label_all, 0.5)

pmsg = 'Test Metrics(best metrics) -> HammingLoss:{0:>5.4f},\n' \
       'Micro/Macro/Instance-based F1:{1:>5.3f}/{2:>5.3f}/{3:>5.3f}, ' \
       'Micro/Macro F1(true number):{4:>5.3f}/{5:>5.3f},\n ' \
       'Micro/Macro/Instance-based Precision:{6:>5.3f}/{7:>5.3f}/{8:>5.3f}, ' \
       'Micro/Macro Precision(true number):{9:>5.3f}/{10:>5.3f},\n ' \
       'Micro/Macro/Instance-based Recall:{11:>5.3f}/{12:>5.3f}/{13:>5.3f}, ' \
       'Micro/Macro Recall(true number):{14:>5.3f}/{15:>5.3f},\n ' \
       'OneError:{16:>5.3f}, Ranking Loss:{17:>5.4f}, Micro/MacroAUC:{18:>5.3f}/{19:>5.3f}, '
print(pmsg.format(test_hl, test_micf1, test_macf1, test_samf1, test_truenum_micf1, test_truenum_macf1,
                  test_micpre, test_macpre, test_sampre, test_truenum_micpre, test_truenum_macpre,
                  test_micrec, test_macrec, test_samrec, test_truenum_micrec, test_truenum_macrec, test_oe,
                  test_rloss, test_microauc, test_macroauc))
print(test_class)
