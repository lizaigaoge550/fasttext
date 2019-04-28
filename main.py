import torch.nn as nn
from torch.utils.data import DataLoader
from help import  *
from tqdm import tqdm
import torch.optim as optim
from modelling import FastText
from data import *
import argparse
from allennlp.nn.util import move_to_device
import os
from pytorch_pretrained_bert import WordpieceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=10)
parser.add_argument('--checkpoint', default='checkpoint')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

bert_weight_path = '../scibert-master/scibert_scivocab_uncased/weights.tar.gz'
bert_vocab =  '../scibert-master/scibert_scivocab_uncased/vocab.txt'


word_vocab = Voc()
#word_vocab('word_vocab.txt')
word_vocab(bert_vocab)
char_vocab = Voc()
char_vocab('char_word_vocab.txt')
print(f'word_vocab len : {len(word_vocab)}')
print(f'char_vocab len : {len(char_vocab)}')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_loss(loss_function, pre_y, y):
    return loss_function(pre_y, y)


def get_accuracy(t, p):
    #print(p)
    _, pred = torch.max(p,dim=-1)
    return (t == pred).long().sum().item()


def collate_fn(batches):
    max_len = 0
    for batch in batches:
        max_len = max(max_len, batch['x'].size(0))
    for batch in batches:
        batch['x'] = torch.cat((batch['x'], torch.LongTensor([0]*(max_len - batch['x'].size(0)))))
    x = torch.stack([batch['x'] for batch in batches])
    y = torch.stack([batch['y'] for batch in batches])
    char_x =torch.stack([batch['char_x'] for batch in batches])
    offset = torch.stack([batch['offset'] for batch in batches])
    char_x_len = torch.stack([batch['char_x_len'] for batch in batches])
    return {'x':x, 'y':y, 'char_x':char_x, 'offset':offset, 'char_x_len':char_x_len}


def valid(model, valid_loader, valid_loss, loss_function, bert=False):
    model.eval()
    valid_accu = 0
    valid_sum = 0
    for batch in tqdm(valid_loader, mininterval=2, desc=' -(Evaling)   ', leave=False):
        batch = move_to_device(batch, 0)
        batch_x = batch['x']
        batch_char_x = batch['char_x']
        if bert:
            batch_x_len = batch['offset']
        else:
            batch_x_len = batch['x_len']
        batch_char_x_len = batch['char_x_len']
        batch_y = batch['y']
        # input, input_lens, char_input, char_input_lens
        predict = model(batch_x, batch_x_len, batch_char_x, batch_char_x_len)
        accu = get_accuracy(batch_y, predict)
        loss = get_loss(loss_function, predict, batch_y.view(-1))
        valid_loss.append(loss.item())
        valid_accu += accu
        valid_sum += predict.size(0)
    avg_valid_loss = sum(valid_loss) / len(valid_loss)
    avg_valid_accu = valid_accu / valid_sum
    return avg_valid_loss, avg_valid_accu

def train(model, opt, loss_function, train_dataset, valid_dataset, is_print_size=False, bert=False):
    epoch = args.epoch
    init_valid_loss = float('inf')
    init_valid_accu = -float('inf')
    while epoch:
        model.train()
        if bert:
            train_loader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=32, shuffle=False, drop_last=True, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
            valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=32, shuffle=False, drop_last=True)
        train_loss = []
        train_accu = 0
        train_sum = 0

        valid_loss = []

        for batch in tqdm(train_loader, mininterval=2, desc=' -(Training)   ', leave=False):
            batch = move_to_device(batch,0)
            batch_x = batch['x']
            batch_char_x = batch['char_x']
            if bert:
                batch_x_len = batch['offset']
            else:
                batch_x_len = batch['x_len']
            batch_char_x_len = batch['char_x_len']
            batch_y = batch['y']
            if is_print_size:
                print(batch_x)
                print_size(batch_x=batch_x, batch_char_x=batch_char_x, batch_x_len=batch_x_len,
                           batch_char_x_len=batch_char_x_len, batch_y=batch_y)
                is_print_size = False
            #input, input_lens, char_input, char_input_lens
            predict = model(batch_x, batch_x_len, batch_char_x, batch_char_x_len)
            loss = get_loss(loss_function, predict, batch_y.view(-1))
            accu = get_accuracy(batch_y, predict)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            train_accu += accu
            train_sum += predict.size(0)
        print(f'---------------train loss : {sum(train_loss) / len(train_loss)}--------- train accu : {train_accu / train_sum}')
        epoch -= 1
        _valid_loss, _valid_accu = valid(model, valid_loader, valid_loss, loss_function, bert=bert)
        if _valid_accu > init_valid_accu:
            init_valid_loss = _valid_loss
            init_valid_accu = _valid_accu
            print(f'---------------eval loss : {init_valid_loss} ------eval accu : {init_valid_accu}')
            if not os.path.exists(args.checkpoint):os.mkdir(args.checkpoint)
            torch.save(model.state_dict(), os.path.join(args.checkpoint,'model_{epoch}_{init_valid_loss}_{init_valid_accu}'.format(
                epoch=epoch, init_valid_loss=init_valid_loss, init_valid_accu=init_valid_accu)))


def main():
    word_piece = WordpieceTokenizer(bert_vocab)
    #get dataset
    #train_dataset, test_dataset = get_dataset(word_vocab, char_vocab)
    #vocab_size, embedding_size,char_vocab_size, char_embedding_size, num_filter, ngram_filter_size, num_classes
    train_dataset, test_dataset = get_dataset_bert(word_vocab, char_vocab, word_piece)
    model = FastText(vocab_size=len(word_vocab), embedding_size=128, char_vocab_size=len(char_vocab), \
                     char_embedding_size=50, num_filter=200, ngram_filter_size=[3], num_classes=2, \
                     bert_weight_path=bert_weight_path)
    model.to('cuda')
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()
    train(model, optimizer, loss_function, train_dataset, test_dataset, bert=True, is_print_size=True)

if __name__ == '__main__':
    main()