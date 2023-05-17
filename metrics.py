import torch
import sklearn.metrics as metrics
import utils
import numpy as np


# torch.cuda.set_per_process_memory_fraction(0.5, 0)


def get_metrics(logits, label, thre):
    logits = logits.cpu()
    label = label.cpu()
    logits_np = logits.detach().numpy()
    label_np = label.detach().numpy()

    pred = logits.gt(thre).int()
    for i in range(pred.size(0)):
        if torch.sum(pred[i]) < 2:
            top = utils.topk(pred[i].cpu().numpy(), 2)
            temp = torch.zeros(label.size(1))
            temp[top] = 1
            pred[i] = temp

    pred1 = logits.clone().detach().numpy()
    for k in range(np.size(pred1, axis=0)):
        top = utils.topk(pred1[k], int(torch.sum(label[k]).item()))
        temp = torch.zeros(label.size(1))
        temp[top] = 1
        pred1[k] = temp

    pred1_matrix = logits.clone().detach().numpy()
    for k in range(np.size(pred1_matrix, axis=0)):
        top = utils.topk(pred1_matrix[k], 1)
        temp = torch.zeros(label.size(1))
        temp[top] = 1
        pred1_matrix[k] = temp
    batch_eq_num = (torch.tensor(pred1_matrix) * label).sum().item()
    batch = label.shape[0]
    acc1 = batch_eq_num * 1.0 / batch

    pred3_matrix = logits.clone().detach().numpy()
    for k in range(np.size(pred3_matrix, axis=0)):
        top = utils.topk(pred3_matrix[k], 3)
        temp = torch.zeros(label.size(1))
        temp[top] = 1
        pred3_matrix[k] = temp
    batch_eq_num = (torch.tensor(pred3_matrix) * label).sum().item()
    acc3 = batch_eq_num * 1.0 / batch

    pred5_matrix = logits.clone().detach().numpy()
    for k in range(np.size(pred5_matrix, axis=0)):
        top = utils.topk(pred5_matrix[k], 5)
        temp = torch.zeros(label.size(1))
        temp[top] = 1
        pred5_matrix[k] = temp
    batch_eq_num = (torch.tensor(pred5_matrix) * label).sum().item()
    acc5 = batch_eq_num * 1.0 / batch



    # _, pred_single = torch.sort(logits, dim=1, descending=True)
    # _, label_single = torch.sort(label, dim=1, descending=True)
    # label_single = label_single[:, :1]
    # pred_1 = pred_single[:, :1]
    # pred_3 = pred_single[:, :3]
    # pred_5 = pred_single[:, :5]
    # label_single = label_single.view(-1, 1)
    # correct_pred1 = torch.sum(pred_1 == label_single)
    # acc1 = torch.div(correct_pred1, label.shape[0])
    # correct_pred3 = torch.sum(pred_3 == label_single)
    # acc3 = torch.div(correct_pred3, label.shape[0])
    # correct_pred5 = torch.sum(pred_5 == label_single)
    # acc5 = torch.div(correct_pred5, label.shape[0])

    # pred_single_all = torch.zeros(pred_single.shape[0], pred_single.shape[1])
    # for i in range(pred_single.shape[0]):
    #     pred_single_all[i, pred_1[i].item()] = 1

    return metrics.hamming_loss(label, pred1), \
           metrics.jaccard_score(label, pred1, average='micro'), \
           metrics.f1_score(label, pred, average='micro'), \
           metrics.f1_score(label, pred, average='macro'), \
           metrics.f1_score(label, pred1, average='micro'), \
           metrics.f1_score(label, pred1, average='macro'), \
           metrics.f1_score(label, pred1, average='samples'), \
           metrics.precision_score(label, pred, average='micro'), \
           metrics.precision_score(label, pred, average='macro'), \
           metrics.precision_score(label, pred1, average='micro'), \
           metrics.precision_score(label, pred1, average='macro'), \
           metrics.precision_score(label, pred1, average='samples'), \
           metrics.recall_score(label, pred, average='micro'), \
           metrics.recall_score(label, pred, average='macro'), \
           metrics.recall_score(label, pred1, average='micro'), \
           metrics.recall_score(label, pred1, average='macro'), \
           metrics.recall_score(label, pred1, average='samples'), \
           evaluate_one_error(logits, label), \
           metrics.label_ranking_loss(label_np, logits_np), \
           MacroAveragingAUC(logits_np, label_np), \
           metrics.roc_auc_score(label_np, logits_np, average='micro'), \
           metrics.f1_score(label, pred1, average='micro'), \
           acc1, \
           acc3, \
           acc5, \
           metrics.classification_report(label, pred1)


def evaluate_one_error(predict, truth):
    _, max_label = predict.max(dim=-1)
    max_label = max_label.unsqueeze(-1)
    predict_max = torch.zeros_like(truth).scatter_(dim=-1, index=max_label, value=1)

    batch_eq_num = (predict_max * truth).sum().item()
    batch = truth.shape[0]

    return (batch - batch_eq_num) * 1.0 / batch


def MacroAveragingAUC(outputs, test_target):
    label_num = outputs.shape[1]
    auc = 0
    count = 0
    for i in range(label_num):
        if sum(test_target[:, i]) != 0:
            auc += metrics.roc_auc_score(test_target[:, i], outputs[:, i])
            count += 1
    return auc / count
