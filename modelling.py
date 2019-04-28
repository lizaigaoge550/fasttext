from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.token_embedders import TokenCharactersEncoder
import torch
import torch.nn as nn
import torch.nn.init as init
from allennlp.modules.token_embedders import PretrainedBertEmbedder

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 char_vocab_size, char_embedding_size, num_filter, ngram_filter_size, num_classes, bert_weight_path=False):
        super().__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size)
        init.uniform_(self.char_embedding.weight, -0.1, 0.1)

        if bert_weight_path:
            self.bert = PretrainedBertEmbedder(bert_weight_path)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_size)
            init.uniform_(self.embedding.weight, -0.1, 0.1)
            self.bert = None
        self.cnn_encoder = CnnEncoder(char_embedding_size, num_filters=num_filter,
                                       ngram_filter_sizes=ngram_filter_size)
        self.char_encoder = TokenCharactersEncoder(self.char_embedding, self.cnn_encoder)
        if bert_weight_path:
            embedding_size = 768
        self.linear_layer = nn.Linear(embedding_size+num_filter, num_classes)
        init.xavier_normal_(self.linear_layer.weight)

    def sequence_mask(self, seq_len, max_len):
        batch_size = seq_len.size(0)
        seq_range = torch.arange(max_len, device='cuda').long()
        seq_range = seq_range.unsqueeze(0).expand(batch_size, max_len)

        seq_length = seq_len.unsqueeze(1).expand(batch_size, max_len).long()
        mask = seq_range < seq_length
        return mask.float()

    def char_sequence_mask(self, seq_len, max_word_len):
        batch_size, _seq_len = seq_len.size()
        seq_range = torch.arange(max_word_len, device='cuda').long()
        seq_range = seq_range.view(1,1,max_word_len).expand(batch_size, _seq_len, max_word_len) #batch, seq_len, max_word_len

        seq_length = seq_len.unsqueeze(-1).expand(batch_size, _seq_len, max_word_len).long()
        mask = seq_range < seq_length
        return mask.float()

    def forward(self, input, input_lens, char_input, char_input_lens):
        batch, seq_len = input.size()
        if self.bert:
            type_id = input.new_zeros(input.size()).zero_()
            input_emb = self.bert(input_ids=input.long(),offsets=input_lens,token_type_ids=type_id.long())
        else:
            input_emb = self.embedding(input) # batch, seq_len, emb_dim
        input_lens = input_lens.view(-1)
        mask = self.sequence_mask(input_lens, seq_len)

        batch, seq_len, word_len = char_input.size()
        #char_input_emb = self.char_embedding(char_input) #batch, seq_len, word_len, char_emb_dim
        char_mask = self.char_sequence_mask(char_input_lens, word_len)
        char_input_emb = self.char_encoder(char_input) #batch, seq_len, num_filter

        #concat
        input_emb = torch.cat((input_emb, char_input_emb), dim=-1) #batch, seq_len, (num_filter+emb_dim)
        if not self.bert:
            input_emb *= mask.unsqueeze(-1)

        #avg
        input_emb = torch.mean(input_emb, dim=1) #batch, num_filter+emb_dim

        output = self.linear_layer(input_emb) #batch, num_classes
        return output

