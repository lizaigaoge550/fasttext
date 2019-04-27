from pytorch_pretrained_bert import WordpieceTokenizer
from functools import reduce
import operator
import numpy as np
import pandas as pd
import re
import html
from collections import defaultdict, Counter
import spacy
from torchvision import transforms
import torch

nlp = spacy.load('en')


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 3


    #write to txt
    def write_to_txt(self, vocab_file):
        print(f'****************vocab size : {len(self.index2word)}*****************')
        with open(vocab_file, 'w', encoding='utf-8') as fw:
            for key in self.word2index.keys():
                fw.write(key + '\n')

    #read to txt
    def __call__(self, vocab_file):
        self.word2index = {}
        self.index2word = {}
        with open(vocab_file) as fr:
            for line in fr.readlines():
                word = line.split('\n')[0]
                if word in self.word2index:
                    raise Exception(f"Duplicate words {word}")
                self.word2index[word] = len(self.word2index)
                self.index2word[len(self.word2index)] = word

    def word2id(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index['UNK']

    def id2word(self, id):
        return self.index2word[id]


    def __len__(self):
        return len(self.word2index)



word_vocab = Voc()
char_vocab = Voc()
class_vocab = Voc()


def get_dataset():
    #get text, label
    train_txt, train_label = create_dataset('train.csv')
    test_txt, test_label = create_dataset('test.csv')
    trn = IMDBDataset(zip(train_txt, train_label), transform=transforms.Compose([Tokenizer(),
                                                                                 ToIndex(word_vocab, char_vocab, class_vocab),
                                                                                 ToTensor(max_word_len=, max_seq_len=,
                                                                                          char_padding_idx=char_vocab.word2id('PAD'),
                                                                                          word_padding_idx=word_vocab.word2id('PAD'))
                                                                                 ]))
    tst = IMDBDataset(zip(test_txt, test_label), transform=transforms.Compose([Tokenizer(),
                                                                                 ToIndex(word_vocab, char_vocab,
                                                                                         class_vocab),
                                                                                 ToTensor(max_word_len=, max_seq_len=,
                                                                                          char_padding_idx=char_vocab.word2id(
                                                                                              'PAD'),
                                                                                          word_padding_idx=word_vocab.word2id(
                                                                                              'PAD'))
                                                                                 ]))
    return trn, tst




class IMDBDataset():
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = {'x':sample[0], 'y':sample[1]}
        sample = self.transform(sample)
        return sample

class Tokenizer():
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', tokenizer=None):
        self.tokenizer = tokenizer
        self.filter = filters

    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        split_x = list(filter(lambda a: a.text.lower() not in self.filter, nlp(x).noun_chunks))
        char_x = [list(word) for word in split_x]
        if isinstance(self.tokenizer, WordpieceTokenizer):
            split_x = list(reduce(operator.add, [self.tokenizer(word.lower()) for word in x.split() if word not in self.filter]))
        sample = {'char_x':char_x, 'x':split_x, 'y':y}
        return sample

class ToIndex():
    def __init__(self, word_vocab, char_vocab, class_vocab):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.class_vocab =  class_vocab
    def __call__(self, sample):
        char_x, x, y = sample['char_x'], sample['x'], sample['y']
        char_x = [[self.char_vocab.word2id(c) for c in word] for word in char_x]
        char_x_len = [len(word) for word in char_x]

        x = [self.word_vocab.word2id(word) for word in x]
        x_len = len(x)

        y = self.char_vocab.word2id(y)

        sample = {'char_x':torch.from_numpy(char_x), 'char_x_len':torch.IntTensor(char_x_len),
                  'x':torch.from_numpy(x), 'x_len':torch.IntTensor(x_len), 'y':torch.IntTensor(y)}
        return sample


class ToTensor():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sample):
        char_x, char_x_len, x, x_len, y = sample['char_x'], sample['char_x_len'], sample['x'], sample['x_len'], sample['y']

        max_word_len = self.kwargs['max_word_len']
        max_seq_len = self.kwargs['max_seq_len']
        char_padding_idx = self.kwargs['char_padding_idx']
        word_padding_idx = self.kwargs['word_padding_idx']


        #padding
        new_x = np.zeros(max_seq_len) * word_padding_idx
        if x_len < max_seq_len:
            new_x[:x_len] = np.array(x)
            char_x_len += [0] * (max_seq_len - x_len)
        else:
            new_x = np.array(x[:max_seq_len])
            x_len = max_seq_len
            char_x_len = char_x_len[:max_seq_len]

        assert len(char_x_len) == max_seq_len

        char_new_x = np.zeros((max_seq_len, max_word_len)) * char_padding_idx
        char_new_x_len = []
        for i, l in enumerate(char_x_len):
            if l:
                if l >= max_word_len:
                    char_new_x[i,:] = char_x[i, :max_word_len]
                    char_new_x_len.append(max_word_len)
                elif l < max_word_len:
                    char_new_x[i,:l] = char_x[i, :]
                    char_new_x_len.append(l)
            else:
                char_new_x_len.append(0)
        assert len(char_new_x_len) == len(char_x_len)
        sample = {'x':new_x, 'x_len':x_len,'char_x':char_new_x, 'char_x_len': char_new_x_len, 'y':y}
        return sample



def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')

    re1 = re.compile(r' +')
    x = re1.sub(' ', html.unescape(x))
    x = re.sub('([:,.,?,!,",`,(,)])',' \1',x)
    return x

def create_dataset(data_path):
    df_test = pd.read_csv(data_path, header=None, chunksize=20000)
    text = []
    label = []
    for i, r in enumerate(df_test):
        labels = r.iloc[:, range(1)].values.astype(np.int)
        texts = r[1].astype(str)
        texts = texts.apply(fixup).values
        text += texts.tolist()
        label += labels.tolist()
    return text, label

def create_vocab(texts, min_count=2):
    print(texts[0])
    word_vocab = open('word_vocab.txt','w',encoding='utf-8')
    char_vocab = open('char_word_vocab.txt','w',encoding='utf-8')
    word_dict = defaultdict(int)
    char_dict = defaultdict(int)

    for text in texts:
        for word in list(nlp(text).noun_chunks):
            word = word.text
            if word not in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n':
                word = word.lower()
                word_dict[word] += 1
                for char in word:
                    char_dict[char] += 1
    print(f'....word len:{len(word_dict)}.......')
    print(f'....char len:{len(char_dict)}.......')
    word_vocab.write('PAD\n'); word_vocab.write('UNK\n')
    for item in Counter(word_dict).most_common():
        if item[1] >= min_count:
            word_vocab.write(item[0]+'\n')
    word_vocab.close()
    char_vocab.write('PAD\n');char_vocab.write('UNK\n')
    for item in Counter(char_dict).most_common():
        char_vocab.write(item[0]+'\n')
    char_vocab.close()



if __name__ == '__main__':
    text, *_ = create_dataset('train.csv')
    create_vocab(text, min_count=2)




