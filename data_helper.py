import os
import torch


class DataHelper(object):
    def __init__(self, dataset, mode, vocab=None):
       
        self.dataset = dataset

        self.mode = mode

        self.base = 'data'

        self.current_set = os.path.join(self.base, 'AAPD/data/', 'aapd_' + mode + '.tsv')  # init this value

        if self.dataset == 'aapd':
            self.current_set = os.path.join(self.base, 'AAPD/data/', 'aapd_' + mode + '.tsv')
        elif self.dataset == 'reuters':
            self.current_set = os.path.join(self.base, 'reuters21578/data/', 'reuters_' + mode + '.tsv')
        elif self.dataset == 'csc':
            self.current_set = os.path.join(self.base, 'CSC/data/', 'csc_' + mode + '.tsv')
        elif self.dataset == 'csj':
            self.current_set = os.path.join(self.base, 'CSJ/data/', 'csj_' + mode + '.tsv')

        content, label = self.get_content()

        label = [list(i) for i in label]
        label = [(list(map(int, i))) for i in label]
        self.label = label

        if vocab is None:
            self.vocab = []
            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

        for i in range(len(self.content) - 1, -1, -1):
            for word in self.content[i]:
                if word == 0:
                    self.content[i].remove(word)
            if len(self.content[i]) == 0 or (len(self.content[i]) == 1 and self.content[i][0] == 0):
                self.content.pop(i)
                self.label.pop(i)

    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            temp = [line.split('\t') for line in all.split('\n')]

        label, content = zip(*temp)

        return content, label

    def word2id(self, word):
        try:
            result = self.d[word.lower()]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                yield content, torch.tensor(label).cuda(), i


if __name__ == '__main__':
    data_helper = DataHelper(dataset='aapd', mode='train')

