import dgl
import torch
import utils
import copy
import torch.nn.functional as F
import numpy as np
import gensim
import math
# torch.cuda.set_per_process_memory_fraction(0.5, 0)


class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 label_embed,
                 islabel_node,
                 iscons,
                 pmi_matrix,
                 vocab,
                 n_gram,
                 drop_out,
                 T,
                 direction,
                 perturbation,
                 edges_matrix,
                 edges_num,
                 max_length=300,
                 cuda=True,
                 ):
        super(Model, self).__init__()

        self.T = T
        self.direction = direction
        self.is_cuda = cuda
        self.iscons = iscons
        self.perturbation = perturbation
        self.vocab = vocab
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        print(edges_num)

        self.node_hidden = torch.nn.Embedding(len(vocab), 300, max_norm=5)  # 75 / 16
        self.node_eta = torch.nn.Embedding.from_pretrained(torch.rand(len(vocab), 1), freeze=False)
        self.label_embed = torch.tensor(label_embed).cuda()
        self.label_node_eta = torch.nn.Embedding.from_pretrained(torch.rand(np.size(self.label_embed, 0), 1),
                                                                 freeze=False)
        self.islabel_node = islabel_node
        self.edges_num = edges_num
        self.pmi_matrix = pmi_matrix
        self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.rand(edges_num, 1), freeze=False)
        self.seq_edge_w.weight.requires_grad = False

        self.hidden_size_node = hidden_size_node
        self.embed_matrix = torch.tensor(self.load_word2vec('~/MLGCN/glove.6B.300d.txt'))
        self.node_hidden.weight.data.copy_(self.embed_matrix)
        self.node_hidden.weight.requires_grad = False # todo: revise

        self.len_vocab = len(vocab)

        self.ngram = n_gram

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.max_length = max_length

        self.edges_matrix = edges_matrix
        self.dropout = torch.nn.Dropout(p=drop_out, inplace=True)
        self.bn = torch.nn.BatchNorm1d(num_features=self.hidden_size_node, momentum=0.1,
                                       affine=True, track_running_stats=True)
        self.activation = torch.nn.ReLU(inplace=True)
        self.Linear = torch.nn.Linear(hidden_size_node, class_num, bias=True)

        self.sigmoid = torch.nn.Sigmoid()

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def load_word2vec(self, word2vec_file):
        # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        # we transfer 'glove' format to 'word2vec' with the below code
        # if you want to retry this project must run this section

        # from gensim.test.utils import datapath, get_tmpfile
        # from gensim.models import KeyedVectors
        # # 输入文件
        # glove_file = word2vec_file
        # # 输出文件
        # tmp_file = "glove2word2vec.txt"
        #
        # # 开始转换
        # from gensim.scripts.glove2word2vec import glove2word2vec
        # glove2word2vec(glove_file, tmp_file)

        # 加载转化后的文件
        model = gensim.models.KeyedVectors.load_word2vec_format('glove2word2vec.txt', binary=False)

        embedding_matrix = []

        for word in self.vocab:
            try:
                embedding_matrix.append(model[word])
            except KeyError:
                embedding_matrix.append(np.random.uniform(-0.1, 0.1, 300))

        embedding_matrix = np.array(embedding_matrix)

        return embedding_matrix

    def add_seq_edges(self, doc_ids, old_to_new: dict):
        edges = []
        old_edge_id = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]

            if self.direction == 'forward':
                fore = index
                back = min(index + self.ngram + 1, len(doc_ids))
            elif self.direction == 'backward':
                fore = max(0, index - self.ngram)
                back = index + 1
            else:  # bidirectional
                fore = max(0, index - self.ngram)
                back = min(index + self.ngram + 1, len(doc_ids))

            for i in range(fore, back):  # construct edges
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]
                # edge is without direction
                edges.append([src, dst])
                # - then get the hidden from parent_graph
                try:
                    old_edge_id.append(self.edges_matrix[(src_word_old, dst_word_old)])
                except KeyError:
                    old_edge_id.append(np.random.randint(0, self.edges_num))

        return edges, old_edge_id

    def seq_to_graph(self, doc_ids, label, mode) -> dgl.DGLGraph():
        # construct a graph for one sentence/sample
        # doc_ids = torch.tensor(doc_ids).cuda()
        global label_edge_weight_begin, label_edge_weight, add_label_node
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        local_vocab = set(doc_ids)
        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))  # old index to this sentence index
        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph().to('cuda')

        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)
        local_node_eta = self.node_eta(local_vocab)
        if mode == 'train' and self.islabel_node == 1:
            add_label_node = 0
            label_node_num = 0
            sub_graph.add_nodes(sum(label))
            for i in range(len(label)):
                if label[i] == 1:
                    local_node_hidden = torch.cat((local_node_hidden, self.label_embed[i].view(1, -1)), dim=0)
                    local_node_eta = torch.cat((local_node_eta,
                                                self.label_node_eta(torch.tensor(i).cuda()).view(1, -1)), dim=0)
                    add_label_node += 1
            for i in range(0, len(label), 2):
                sub_graph.add_nodes(1)
                local_node_hidden = torch.cat((local_node_hidden, self.label_embed[i].view(1, -1)), dim=0)
                local_node_eta = torch.cat((local_node_eta,
                                                self.label_node_eta(torch.tensor(i).cuda()).view(1, -1)), dim=0)
                add_label_node += 1
        if self.islabel_node == 2 or self.islabel_node == 3 or (mode != 'train' and self.islabel_node == 1):
            sub_graph.add_nodes(len(self.label_embed))
            for i in range(len(self.label_embed)):
                local_node_hidden = torch.cat((local_node_hidden, self.label_embed[i].view(1, -1)), dim=0)
                local_node_eta = torch.cat((local_node_eta,
                                            self.label_node_eta(torch.tensor(i).cuda()).view(1, -1)), dim=0)
        sub_graph.ndata['h'] = local_node_hidden
        sub_graph.ndata['eta'] = local_node_eta
        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new)
        edges, old_edge_id = [], []
        # add label-label and label-word edge
        edges.extend(seq_edges)
        old_edge_id.extend(seq_old_edges_id)
        label_edge_num = 0
        if mode == 'train' and self.islabel_node == 1:
            for i in range(sum(label)):
                if label[i] == 1:
                    if len(local_vocab) > self.ngram:
                        num_gram = self.ngram
                    else:
                        num_gram = len(local_vocab)
                    selected_node = np.random.choice(range(0, len(local_vocab)), size=num_gram,
                                                     replace=False)
                    for j in selected_node:
                        edges.append([len(local_vocab) + i, j])  # edge label -> word
                        label_edge_num = label_edge_num + 1

            for j in range(len(local_vocab), len(local_vocab) + add_label_node):
                for k in range(len(local_vocab), len(local_vocab) + sum(label)):
                    edges.append([j, k])
                    label_edge_num = label_edge_num + 1
        '''
        contruct edges for label-label and label-word
        islabel_node = 2 : do not connect label-label edges, only connect label-word and random choose word to connect
        islabel_node = 3 : connect label-label and label-word, label-label edges are connected with positive pmi
        '''

        if self.islabel_node == 2:
            for i in range(len(self.label_embed)):
                if len(local_vocab) > self.ngram:
                    num_gram = self.ngram
                else:
                    num_gram = len(local_vocab)
                selected_node = np.random.choice(range(0, len(local_vocab)), size=num_gram,
                                                 replace=False)
                for j in selected_node:
                    edges.append([len(local_vocab) + i, j])  # edge label -> word
                    label_edge_num = label_edge_num + 1

        elif (self.islabel_node == 3 and mode != 'train') or (mode != 'train' and self.islabel_node == 1):  # origin
            for i in range(len(self.label_embed)):
                if len(local_vocab) > self.ngram:
                    num_gram = self.ngram
                else:
                    num_gram = len(local_vocab)
                selected_node = np.random.choice(range(0, len(local_vocab)), size=num_gram,
                                                 replace=False)
                for j in selected_node:
                    edges.append([len(local_vocab) + i, j])  # edge label -> word
                    label_edge_num = label_edge_num + 1
            label_edge_weight_begin = label_edge_num
            label_edge_weight = list()
            for j in range(len(self.label_embed)):
                topk_index = utils.topk(self.pmi_matrix[j], self.ngram)
                for k in topk_index:
                    edges.append([len(local_vocab) + j, len(local_vocab) + k])
                    label_edge_num = label_edge_num + 1
                    label_edge_weight.append(self.pmi_matrix[j][k])

        elif self.islabel_node == 3 and mode == 'train':
            for i in range(sum(label)):
                if label[i] == 1:
                    if len(local_vocab) > self.ngram:
                        num_gram = self.ngram
                    else:
                        num_gram = len(local_vocab)
                    selected_node = np.random.choice(range(0, len(local_vocab)), size=num_gram,
                                                     replace=False)
                    for j in selected_node:
                        edges.append([len(local_vocab) + i, j])  # edge label -> word
                        label_edge_num = label_edge_num + 1

            label_edge_weight_begin = label_edge_num
            label_edge_weight = list()
            for j in range(len(self.label_embed)):
                topk_index = utils.topk(self.pmi_matrix[j], self.ngram)
                for k in topk_index:
                    edges.append([len(local_vocab) + j, len(local_vocab) + k])
                    label_edge_num = label_edge_num + 1
                    label_edge_weight.append(self.pmi_matrix[j][k])

        # seq_edge is edge between index in dic old_to_new
        # seq_old_edges_id is edge old
        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)

        srcs, dsts = zip(*edges)
        sub_graph.add_edges(srcs, dsts)

        seq_edges_w = self.seq_edge_w(old_edge_id)
        if self.islabel_node == 2 or (mode == 'train' and self.islabel_node == 1):
            seq_edges_w = torch.cat((seq_edges_w, (torch.rand(label_edge_num, 1)).cuda()), dim=0)
        elif (mode != 'train' and self.islabel_node == 1) or self.islabel_node == 3:
            seq_edges_w = torch.cat((seq_edges_w, (torch.rand(label_edge_weight_begin, 1)).cuda()), dim=0)
            label_edge_weight = torch.tensor(label_edge_weight).view(-1, 1).cuda()
            seq_edges_w = torch.cat((seq_edges_w, label_edge_weight), dim=0)
        sub_graph.edata['w'] = seq_edges_w  # edges weight

        return sub_graph

    def seq_to_graph_node(self, sub_graph, doc_ids: list, mode) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]
        doc_set = set(doc_ids)
        length = len(doc_set)
        num = int(length * 0.2)
        selected_node = np.random.choice(range(0, length), size=num, replace=False)
        selected_node = sorted(selected_node, reverse=True)
        for i in selected_node:
            sub_graph.remove_nodes(i)
        return sub_graph

    def seq_to_graph_shuffle(self, sub_graph, doc_ids: list, mode) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]
        doc_set = set(doc_ids)
        length = len(doc_set)
        num = int(length * 0.2)
        selected_node1 = np.random.choice(range(0, length), size=num, replace=False)
        selected_node2 = np.random.choice(range(0, length), size=num, replace=False)
        for i in selected_node1:
            for j in selected_node2:
                if i != j:
                    temp = sub_graph.ndata['h'][i].clone()
                    sub_graph.ndata['h'][i] = sub_graph.ndata['h'][j]
                    sub_graph.ndata['h'][j] = temp
                    temp = sub_graph.ndata['eta'][i].clone()
                    sub_graph.ndata['eta'][i] = sub_graph.ndata['eta'][j]
                    sub_graph.ndata['eta'][j] = temp
        return sub_graph

    def seq_to_graph_edge(self, sub_graph, doc_ids: list, label,  perturbation, mode) -> dgl.DGLGraph():
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]
        doc_set = set(doc_ids)
        if len(doc_set) > self.ngram:
            num_gram = self.ngram
        else:
            num_gram = len(doc_set)
        #if self.islabel_node != 0:
        #    length = (sub_graph.number_of_edges() - torch.sum(label) * num_gram - label.shape[0] * self.ngram)
        #else:
        length = sub_graph.number_of_edges()
        num = math.ceil(length * 0.2)
        # if perturbation == 'delete':
        selected_edge = np.random.choice(range(0, length), size=num, replace=False)
        selected_edge = sorted(selected_edge, reverse=True)
        for i in selected_edge:
            sub_graph.edata['w'][i] = torch.tensor([0])


        return sub_graph

    def gcn_msg(self, edge):
        return {'m': edge.src['h'], 'w': edge.data['w']}

    def gcn_reduce(self, node):
        w = node.mailbox['w']
        new_hidden = torch.mul(w, node.mailbox['m'])
        new_hidden, _ = torch.max(new_hidden, 1)
        node_eta = node.data['eta']
        new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
        return {'h': new_hidden}

    def forward(self, doc_ids, label, mode):

        if (self.islabel_node == 1 and mode == 'train') or (self.islabel_node == 3 and mode == 'train'):
            sub_graphs = list()
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph(doc_ids[i], label[i], mode))
        else:
            sub_graphs = [self.seq_to_graph(doc, label, mode) for doc in doc_ids]

        # sub_graphs = [self.seq_to_graph(doc, mode) for doc in doc_ids]
        if self.iscons == 1 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))

            sub_graphs_cons = list()
            for i in range(len(doc_ids)):
                sub_graphs_cons.append(self.seq_to_graph_node(sub_graphs_copy[i], doc_ids[i], mode))
            sub_graphs.extend(sub_graphs_cons)

        elif self.iscons == 2 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))

            sub_graphs_cons = list()
            for i in range(len(doc_ids)):
                sub_graphs_cons.append(self.seq_to_graph_edge(sub_graphs_copy[i], doc_ids[i], label[i], self.perturbation, mode))
            sub_graphs.extend(sub_graphs_cons)

        elif self.iscons == 3 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))

            sub_graphs_cons = list()
            for i in range(len(doc_ids)):
                sub_graphs_cons.append(self.seq_to_graph_shuffle(sub_graphs_copy[i], doc_ids[i], mode))
            sub_graphs.extend(sub_graphs_cons)

        elif self.iscons == 12 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))
            sub_graphs = list()
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_node(sub_graphs_copy[i], doc_ids[i], mode))
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_edge(sub_graphs_copy[i], doc_ids[i], label[i], self.perturbation, mode))

        elif self.iscons == 13 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))
            sub_graphs = list()
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_node(sub_graphs_copy[i], doc_ids[i], mode))
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_shuffle(sub_graphs_copy[i], doc_ids[i], mode))

        elif self.iscons == 23 and mode == 'train':
            sub_graphs_copy = list()
            for i in sub_graphs:
                sub_graphs_copy.append(copy.copy(i))
            sub_graphs = list()
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_edge(sub_graphs_copy[i], doc_ids[i], label[i], self.perturbation, mode))
            for i in range(len(doc_ids)):
                sub_graphs.append(self.seq_to_graph_shuffle(sub_graphs_copy[i], doc_ids[i], mode))

        batch_graph = dgl.batch(sub_graphs)

        for k in range(self.T):
            batch_graph.update_all(self.gcn_msg, self.gcn_reduce)

        h1 = dgl.sum_nodes(batch_graph, feat='h')
        h1 = self.dropout(h1)
        feature = self.activation(h1)  # variables to participate in contrastive learning

        logits = self.Linear(feature)
        
        pre = self.sigmoid(logits)

        return feature, pre
