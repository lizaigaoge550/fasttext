import torch
import torch.nn as nn
from data import get_dataset
from torch.utils.data import DataLoader
from help import  *
from tqdm import tqdm
import torch.optim as optim
from modelling import FastText

def train(train_dataset, valid_dataset, is_print_size=False):
    while epoch:
        train_loader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, num_workers=4, batch_size=32, shuffle=False, drop_last=True)
        for batch in tqdm(train_loader, mininterval=2, desc=' -(Training)   ', leave=False):
            batch_x = batch['x']
            batch_char_x = batch['char_x']
            batch_x_len = batch['x_len']
            batch_char_x_len = batch['char_x_len']
            batch_y = batch['y']
            if is_print_size:
                print_size(batch_x=batch_x, batch_char_x=batch_char_x, batch_x_len=batch_x_len, batch_char_x_len=batch_char_x_len, batch_y=batch_y)
                is_print_size = False




def main():
    #get dataset
    train_dataset, test_dataset = get_dataset()
    #vocab_size, embedding_size,char_vocab_size, char_embedding_size, num_filter, ngram_filter_size, num_classes
    model = FastText()
    optimizer = optim.Adam()

    train(train_dataset, test_dataset)
