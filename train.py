import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from model import Model
from CLoss import ContrastiveLoss
from data_helper import DataHelper
from dataset import MyDataset
import utils
import numpy as np
import tqdm
import math
# import sys
import random
import argparse
import time
import datetime
import metrics
import os
import warnings

warnings.filterwarnings('ignore')
# torch.cuda.set_per_process_memory_fraction(0.5, 0)

NUM_ITER_EVAL = 500  # 500
PRINT_ITER = 500  # 500
TEST_ITER = 1000  # 1000
EARLY_STOP_TIME = 5


def collate_fn(batch):
    #  redefine collate_fn
    batch = list(zip(*batch))
    labels = torch.tensor(batch[1])
    texts = list(batch[0])
    return texts, labels


def count_unique_label(y_true, y_pred):
    unique_label = []
    for each in y_true:
        if each not in unique_label:
            unique_label.append(each)
    for i in y_pred:
        for j in i:
            if j not in unique_label:
                unique_label.append(j)
    unique_label = list(set(unique_label))
    return unique_label


def edges_mapping(vocab_len, content, ngram):
    d = {}
    count = 1
    zero = 0

    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i - ngram), min(len(doc), i + ngram + 1)):
                dst = doc[dst_id]
                '''
                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                '''
                if count == 1:
                    d[(src, dst)] = count
                    count += 1
                else:
                    if (src, dst) not in d.keys():
                        d[(src, dst)] = count
                        count += 1
                        if src == 0 or dst == 0:
                            zero += 1

    print('zero : %d, poration: %.2f' % (zero, zero / count))
    return count, d


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset, dev_data_helper, dev_dataloader, thre):
    model.eval()
    iter = 0
    pred_all = torch.tensor([]).cuda()
    label_all = torch.tensor([]).cuda()
    with torch.no_grad():
        # for content, label, _ in dev_data_helper.batch_iter(batch_size=128, num_epoch=1):
        for content, label in dev_dataloader:
            iter += 1
            label = label.cuda()
            label_emp = []
            feature, logits = model(content, label_emp, mode='dev/test')
            pred_all = torch.cat((pred_all, logits), dim=0)
            label_all = torch.cat((label_all, label), dim=0)

    dev_hl, dev_jac, dev_micf1, dev_macf1, dev_truenum_micf1, dev_truenum_macf1, dev_samf1, dev_micpre, dev_macpre, \
    dev_truenum_micpre, dev_truenum_macpre, dev_sampre, \
    dev_micrec, dev_macrec, dev_truenum_micrec, dev_truenum_macrec, dev_samrec, dev_oe, dev_rloss, \
    dev_macroauc, dev_microauc, f1_single, acc1_single, acc3_single, acc5_single, _ = metrics.get_metrics(pred_all,
                                                                                                          label_all,
                                                                                                          thre)

    return dev_hl, dev_jac, dev_micf1, dev_macf1, dev_truenum_micf1, dev_truenum_macf1, dev_samf1, dev_micpre, dev_macpre, \
           dev_truenum_micpre, dev_truenum_macpre, dev_sampre, \
           dev_micrec, dev_macrec, dev_truenum_micrec, dev_truenum_macrec, dev_samrec, dev_oe, dev_rloss, \
           dev_macroauc, dev_microauc, f1_single, acc1_single, acc3_single, acc5_single


def test(model, dataset, thre, batch):
    model.cuda()
    data_helper = DataHelper(dataset=dataset, mode='test')
    test_dataset = MyDataset(data_helper)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=16,
                                 pin_memory=False, collate_fn=collate_fn)
    iter = 0
    pred_all = torch.tensor([]).cuda()
    label_all = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for content, label in test_dataloader:
            iter += 1
            label = label.cuda()
            label_emp = []
            feature, logits = model(content, label_emp, mode='dev/test')

            pred_all = torch.cat((pred_all, logits), dim=0)
            label_all = torch.cat((label_all, label), dim=0)

    test_hl, test_jac, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, test_micpre, test_macpre, \
    test_truenum_micpre, test_truenum_macpre, test_sampre, \
    test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
    test_macroauc, test_microauc, f1_single, acc1_single, acc3_single, acc5_single, test_class = metrics.get_metrics(
        pred_all, label_all, thre)

    return test_hl, test_jac, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, \
           test_micpre, test_macpre, test_truenum_micpre, test_truenum_macpre, test_sampre, \
           test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
           test_macroauc, test_microauc, f1_single, acc1_single, acc3_single, acc5_single, test_class


def train(num_epoch, ngram, name, bar: object, drop_out, dataset, is_cuda, hidden_node_size,
          A, B, islabel_node, thre, T, direction, iscons, batch, temperature, lambdafactor, perturbation):
    global supervisedloss, contrastiveloss, pbar
    print('Data Loading.')
    with open('data/vocab.txt', 'r') as f:
        vocab = f.read()
        vocab = vocab.split('\n')

    '''
    data_helper.content is the vocabulary index of every sample
    data_helper.d is the vocabulary index
    data_helper.label is the label of every sample 
        (note that it is the label index, true label is in data_helper.label_str)
    '''
    data_helper = DataHelper(dataset=dataset, mode='train', vocab=vocab)
    train_dataset = MyDataset(data_helper)
    dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=16,
                            pin_memory=False, collate_fn=collate_fn, drop_last=True)
    label_embed = [np.random.uniform(-0.1, 0.1, 300).astype(np.float32) for i in range(len(data_helper.label[0]))]
    # edge_mapping is the constructed graph
    count, edges_mappings = edges_mapping(len(data_helper.vocab), data_helper.content, ngram)

    co_matrix = utils.get_pmi(data_helper.label)

    model = Model(class_num=len(data_helper.label[0]), hidden_size_node=hidden_node_size, label_embed=label_embed,
                  islabel_node=islabel_node, iscons=iscons, pmi_matrix=co_matrix,
                  vocab=data_helper.vocab, n_gram=ngram, drop_out=drop_out, T=T, edges_matrix=edges_mappings,
                  direction=direction, perturbation=perturbation, edges_num=count, cuda=is_cuda)

    dev_data_helper = DataHelper(dataset=dataset, mode='dev', vocab=vocab)
    dev_dataset = MyDataset(dev_data_helper)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch, shuffle=True, num_workers=16,
                                pin_memory=False, collate_fn=collate_fn)
    print(model)
    if is_cuda:
        print('cuda')
        model.cuda()
    # loss_func = torch.nn.CrossEntropyLoss()  #  CE is not suitable for multi-label

    pos_weight = utils.get_class_weight(data_helper.label, A, B).cuda()
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    closs = ContrastiveLoss(batch_size=batch, temperature=temperature)

    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8, verbose=True)

    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_metrics = 0.0
    last_best_time = 0
    start_time = time.time()

    now = 0
    clhistory = list()
    suphistory = list()
    sup1 = list()
    sup2 = list()
    # for content, label, epoch in data_helper.batch_iter(batch_size=128, num_epoch=100):  # epoch = 100
    for epoch in range(num_epoch):  # epoch = 100
        for content, label in dataloader:
            improved = ''
            model.train()
            label = label.cuda()

            feature, logits = model(content, label, mode='train')

            if iscons != 0:
                feature1 = feature[0:int(feature.shape[0] / 2), :]
                feature2 = feature[int(feature.shape[0] / 2):feature.shape[0], :]
                suploss1 = loss_func(feature1, label.float())
                suploss2 = loss_func(feature2, label.float())
                label = torch.cat((label, label), dim=0)
                supervisedloss = loss_func(feature, label.float())
                contrastiveloss = closs(feature1, feature2)
                loss = supervisedloss + contrastiveloss * lambdafactor

                clhistory.append(contrastiveloss.item())
                suphistory.append(supervisedloss.item())
                sup1.append(suploss1.item())
                sup2.append(suploss2.item())
            else:
                loss = loss_func(feature, label.float())

            # BCELoss should input feature instead of logits
            optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optim.step()

            iter += 1

            if bar:
                pbar.update()

            if iter % NUM_ITER_EVAL == 0:
                # if iter >= 250 * 5:
                #     scheduler.step()
                train_hl, train_jac, train_micf1, train_macf1, train_truenum_micf1, train_truenum_macf1, train_samf1, \
                train_micpre, train_macpre, train_truenum_micpre, train_truenum_macpre, \
                train_sampre, train_micrec, train_macrec, train_truenum_micrec, train_truenum_macrec, train_samrec, \
                train_oe, train_rloss, train_macroauc, train_microauc, f1_single, acc1_single, acc3_single, acc5_single, _ = metrics.get_metrics(
                    logits, label, thre)
                if bar:
                    pbar.close()
                print('Training Metrics-> HammingLoss:%.4f,JaccordIndex:%.4f \n'
                      'Micro/Macro/Instance-based F1:%.3f/%.3f/%.3f, Micro/Macro F1(true number):%.3f/%.3f, \n'
                      'Micro/Macro/Instance-based Precision:%.3f/%.3f/%.3f, '
                      'Micro/Macro Precision(true number):%.3f/%.3f, \n'
                      'Micro/Macro/Instance-based Recall:%.3f/%.3f/%.3f, '
                      'Micro/Macro Recall(true number):%.3f/%.3f, \n'
                      'OneError:%.3f, Ranking Loss:%.4f, Micro/MacroAUC:%.3f/%.3f, \n'
                      'F1 (single):%.3f, acc1 (single):%.3f, acc3 (single):%.3f, acc5 (single):%.3f'
                      % (train_hl, train_jac, train_micf1, train_macf1, train_samf1, train_truenum_micf1,
                         train_truenum_macf1,
                         train_micpre, train_macpre, train_sampre, train_truenum_micpre, train_truenum_macpre,
                         train_micrec, train_macrec, train_samrec, train_truenum_micrec, train_truenum_macrec, train_oe,
                         train_rloss, train_microauc, train_macroauc, f1_single, acc1_single, acc3_single, acc5_single))
                print('loss:%.4f, supervised loss:%.4f, contrastive loss:%.4f' % (loss, supervisedloss, contrastiveloss))
                if bar:
                    pbar = tqdm.tqdm(total=NUM_ITER_EVAL)

            if iter % PRINT_ITER == 0:
                now += 1
                dev_hl, dev_jac, dev_micf1, dev_macf1, dev_truenum_micf1, dev_truenum_macf1, dev_samf1, dev_micpre, dev_macpre, \
                dev_truenum_micpre, dev_truenum_macpre, dev_sampre, \
                dev_micrec, dev_macrec, dev_truenum_micrec, dev_truenum_macrec, dev_samrec, dev_oe, dev_rloss, \
                dev_macroauc, dev_microauc, f1_single, acc1_single, acc3_single, acc5_single = dev(model, dataset,
                                                                                                   dev_data_helper,
                                                                                                   dev_dataloader, thre)

                if dev_truenum_micf1 + dev_truenum_macf1 >= best_metrics:  # save the best model
                    best_metrics = dev_truenum_micf1 + dev_truenum_macf1
                    last_best_time = now
                    improved = '*'
                    torch.save(model.state_dict(), name + '_best.pth')

                if now - last_best_time >= EARLY_STOP_TIME:
                    print('\n ALREADY HAVE A GOOD MODEL\n')

                    return model
                msg = 'Epoch: {0:>6} Iter: {1:>6}, DevMetrics-> HammingLoss:{2:>5.4f}, JaccordIndex:{3:>5.4f} \n' \
                      'Micro/Macro/Instance-based F1:{4:>5.3f}/{5:>5.3f}/{6:>5.3f}, ' \
                      'Micro/Macro F1(true number):{7:>5.3f}/{8:>5.3f}, \n' \
                      'Micro/Macro/Instance-based Precision:{9:>5.3f}/{10:>5.3f}/{11:>5.3f}, ' \
                      'Micro/Macro Precision(true number):{12:>5.3f}/{13:>5.3f}, \n' \
                      'Micro/Macro/Instance-based Recall:{14:>5.3f}/{15:>5.3f}/{16:>5.3f}, ' \
                      'Micro/Macro Recall(true number):{17:>5.3f}/{18:>5.3f}, \n' \
                      'OneError:{19:>5.3f}, Ranking Loss:{20:>5.4f}, Micro/MacroAUC:{21:>5.3f}/{22:>5.3f}, \n' \
                      'F1 (single):{23:>5.3f}, acc1 (single):{24:>5.3f}, acc3 (single):{25:>5.3f}, acc5 (single):{26:>5.3f}, \n' \
                      'Time: {27}{28}'
                print(msg.format(epoch, iter, dev_hl, dev_jac, dev_micf1, dev_macf1, dev_samf1,
                                         dev_truenum_micf1, dev_truenum_macf1, dev_micpre, dev_macpre,
                                         dev_sampre, dev_truenum_micpre, dev_truenum_macpre, dev_micrec, dev_macrec,
                                         dev_samrec,
                                         dev_truenum_micrec, dev_truenum_macrec, dev_oe, dev_rloss,
                                         dev_microauc, dev_macroauc, f1_single, acc1_single, acc3_single, acc5_single,
                                         get_time_dif(start_time), improved))
            if iter % TEST_ITER == 0:
                test_hl, test_jac, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, \
                test_micpre, test_macpre, test_truenum_micpre, test_truenum_macpre, test_sampre, \
                test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
                test_macroauc, test_microauc, f1_single, acc1_single, acc3_single, acc5_single, _ = test(model, dataset,
                                                                                                         thre, batch)

                pmsg = 'Test Metrics(final epoch) -> HammingLoss:{0:>5.4f}, JaccordIndex:{1:>5.4f}\n' \
                       'Micro/Macro/Instance-based F1:{2:>5.3f}/{3:>5.3f}/{4:>5.3f}, ' \
                       'Micro/Macro F1(true number):{5:>5.3f}/{6:>5.3f},\n ' \
                       'Micro/Macro/Instance-based Precision:{7:>5.3f}/{8:>5.3f}/{9:>5.3f}, ' \
                       'Micro/Macro Precision(true number):{10:>5.3f}/{11:>5.3f},\n ' \
                       'Micro/Macro/Instance-based Recall:{12:>5.3f}/{13:>5.3f}/{14:>5.3f}, ' \
                       'Micro/Macro Recall(true number):{15:>5.3f}/{16:>5.3f},\n ' \
                       'OneError:{17:>5.3f}, Ranking Loss:{18:>5.4f}, Micro/MacroAUC:{19:>5.3f}/{20:>5.3f}, \n' \
                       'F1 (single):{21:>5.3f}, acc1 (single):{22:>5.3f}, acc3 (single):{23:>5.3f}, acc5 (single):{24:>5.3f}'
                print(pmsg.format(test_hl, test_jac, test_micf1, test_macf1, test_samf1, test_truenum_micf1,
                                  test_truenum_macf1,
                                  test_micpre, test_macpre, test_sampre, test_truenum_micpre, test_truenum_macpre,
                                  test_micrec, test_macrec, test_samrec, test_truenum_micrec, test_truenum_macrec,
                                  test_oe,
                                  test_rloss, test_microauc, test_macroauc, f1_single, acc1_single, acc3_single,
                                  acc5_single))

            torch.cuda.empty_cache()
            # return model
    sup1_pd = pd.DataFrame(data=sup1)
    sup1_pd.to_csv('OriginSuperviseLoss.csv')
    sup2_pd = pd.DataFrame(data=sup2)
    sup2_pd.to_csv('AugmentSuperviseLoss.csv')
    sup_pd = pd.DataFrame(data=suphistory)
    sup_pd.to_csv('SuperviseLoss.csv')
    con_pd = pd.DataFrame(data=clhistory)
    con_pd.to_csv('ContrastiveLoss.csv')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', required=False, type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, default=256, help='batch size')
    parser.add_argument('--hidden_node_size', required=False, type=int, default=300, help='hidden size')
    parser.add_argument('--ngram', required=False, type=int, default=2, help='ngram number')
    parser.add_argument('--T', required=False, type=int, default=2, help='GAT layers number')
    parser.add_argument('--isposw', required=False, type=bool, default=True, help='whether need pos weight')
    parser.add_argument('--islabel_node', required=False, type=int, default=0, help='whether add label node,'
                                                                                    '0-no add,1-all zero,2-random')
    parser.add_argument('--iscons', required=False, type=int, default=0, help='whether add constractive learning,'
                                                                              '0-no add,1-del node,2-del edge')
    parser.add_argument('--perturbation', required=False, type=str, default='delete', help='delete / add edges')
    parser.add_argument('--A', required=False, type=float, default=0.4, help='pos weight parameter A')
    parser.add_argument('--B', required=False, type=float, default=0.5, help='pos weight parameter B')
    parser.add_argument('--direction', required=False, type=str, default='backward', help='graph edge direction')
    parser.add_argument('--name', required=False, type=str, default='temp_model', help='project name')
    parser.add_argument('--bar', required=False, type=int, default=1, help='show bar')
    parser.add_argument('--dropout', required=False, type=float, default=0.7, help='dropout rate')
    parser.add_argument('--dataset', required=False, type=str, default='aapd', help='dataset:aapd/rcv/reuters')
    parser.add_argument('--rand', required=False, type=int, default=7, help='rand_seed')
    parser.add_argument('--threshold', required=False, type=float, default=0.5, help='predict threshold')
    parser.add_argument('--temperature', required=False, type=float, default=2, help='temperature in contrastive loss')
    parser.add_argument('--lambdafactor', required=False, type=float, default=0.02, help='lambda of contrastive loss')

    args = parser.parse_args()

    print('ngram: %d' % args.ngram)
    print('project_name: %s' % args.name)
    print('dataset: %s' % args.dataset)
    # #
    SEED = args.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.bar == 1:
        bar = True
    else:
        bar = False

    model = train(args.num_epoch, args.ngram, args.name, bar, args.dropout, dataset=args.dataset, is_cuda=True,
                  hidden_node_size=args.hidden_node_size, A=args.A, B=args.B, islabel_node=args.islabel_node,
                  thre=args.threshold, T=args.T, direction=args.direction, iscons=args.iscons, batch=args.batch_size,
                  temperature=args.temperature, lambdafactor=args.lambdafactor, perturbation=args.perturbation)

    try:
        torch.save(model, args.dataset + str(args.hidden_node_size) + 'd-ngram=' + str(args.ngram) + '-T=' + str(args.T)
                   + '-direction=' + args.direction +
                   '-dropout=' + str(args.dropout) + '-AB=' + str(args.A) + '-' + str(args.B) +
                   '-label_node=' + str(args.islabel_node) + '-constrative=' + str(args.iscons) +
                   '-temper=' + str(args.temperature) + '-lambda=' + str(args.lambdafactor) + '.pth')
    except RuntimeWarning:
        print('fail to save final model')

    # test the last model
    test_hl, test_jac, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, \
    test_micpre, test_macpre, test_truenum_micpre, test_truenum_macpre, test_sampre, \
    test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
    test_macroauc, test_microauc, f1_single, acc1_single, acc3_single, acc5_single, test_class = test(model, args.dataset, args.threshold, args.batch_size)

    pmsg = 'Test Metrics(final epoch) -> HammingLoss:{0:>5.4f}, JaccordIndex:{1:>5.4f}\n' \
           'Micro/Macro/Instance-based F1:{2:>5.3f}/{3:>5.3f}/{4:>5.3f}, ' \
           'Micro/Macro F1(true number):{5:>5.3f}/{6:>5.3f},\n ' \
           'Micro/Macro/Instance-based Precision:{7:>5.3f}/{8:>5.3f}/{9:>5.3f}, ' \
           'Micro/Macro Precision(true number):{10:>5.3f}/{11:>5.3f},\n ' \
           'Micro/Macro/Instance-based Recall:{12:>5.3f}/{13:>5.3f}/{14:>5.3f}, ' \
           'Micro/Macro Recall(true number):{15:>5.3f}/{16:>5.3f},\n ' \
           'OneError:{17:>5.3f}, Ranking Loss:{18:>5.4f}, Micro/MacroAUC:{19:>5.3f}/{20:>5.3f}, \n' \
           'F1 (single):{21:>5.3f}, acc1 (single):{22:>5.3f}, acc3 (single):{23:>5.3f}, acc5 (single):{24:>5.3f}'
    print(pmsg.format(test_hl, test_jac, test_micf1, test_macf1, test_samf1, test_truenum_micf1, test_truenum_macf1,
                      test_micpre, test_macpre, test_sampre, test_truenum_micpre, test_truenum_macpre,
                      test_micrec, test_macrec, test_samrec, test_truenum_micrec, test_truenum_macrec,
                      test_oe,
                      test_rloss, test_microauc, test_macroauc, f1_single, acc1_single, acc3_single, acc5_single))
    print(test_class)

    # test the the model with the best metrics
    model.state_dict(torch.load(args.name + '_best.pth'))

    test_hl, test_jac, test_micf1, test_macf1, test_truenum_micf1, test_truenum_macf1, test_samf1, \
    test_micpre, test_macpre, test_truenum_micpre, test_truenum_macpre, test_sampre, \
    test_micrec, test_macrec, test_truenum_micrec, test_truenum_macrec, test_samrec, test_oe, test_rloss, \
    test_macroauc, test_microauc, f1_single, acc1_single, acc3_single, acc5_single, test_class = test(model, args.dataset, args.threshold, args.batch_size)

    pmsg = 'Test Metrics(best metrics) -> HammingLoss:{0:>5.4f}, JaccordIndex:{1:>5.4f}\n' \
           'Micro/Macro/Instance-based F1:{2:>5.3f}/{3:>5.3f}/{4:>5.3f}, ' \
           'Micro/Macro F1(true number):{5:>5.3f}/{6:>5.3f},\n ' \
           'Micro/Macro/Instance-based Precision:{7:>5.3f}/{8:>5.3f}/{9:>5.3f}, ' \
           'Micro/Macro Precision(true number):{10:>5.3f}/{11:>5.3f},\n ' \
           'Micro/Macro/Instance-based Recall:{12:>5.3f}/{13:>5.3f}/{14:>5.3f}, ' \
           'Micro/Macro Recall(true number):{15:>5.3f}/{16:>5.3f},\n ' \
           'OneError:{17:>5.3f}, Ranking Loss:{18:>5.4f}, Micro/MacroAUC:{19:>5.3f}/{20:>5.3f}, \n' \
           'F1 (single):{21:>5.3f}, acc1 (single):{22:>5.3f}, acc3 (single):{23:>5.3f}, acc5 (single):{24:>5.3f}'
    print(pmsg.format(test_hl, test_jac, test_micf1, test_macf1, test_samf1, test_truenum_micf1, test_truenum_macf1,
                      test_micpre, test_macpre, test_sampre, test_truenum_micpre, test_truenum_macpre,
                      test_micrec, test_macrec, test_samrec, test_truenum_micrec, test_truenum_macrec,
                      test_oe,
                      test_rloss, test_microauc, test_macroauc, f1_single, acc1_single, acc3_single, acc5_single))
