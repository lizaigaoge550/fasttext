def print_size(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} : {value.size()}')


def tokens_to_indices(tokens, vocab, wordpiece_tokenizer, max_pieces=512):
    wordpiece_ids = [vocab.word2id('[CLS]')]
    offset = 1
    offsets = []
    for token in tokens:
        text = token.lower()
        token_wordpiece_ids = [vocab.word2id(wordpiece) for wordpiece in wordpiece_tokenizer.tokenize(text)]
        if len(wordpiece_ids) + len(token_wordpiece_ids) + 1 <= max_pieces:
            offsets.append(offset)
            offset += len(token_wordpiece_ids)
            wordpiece_ids.extend(token_wordpiece_ids)
        else:
            break
    wordpiece_ids.extend([vocab.word2id('[SEP]')])
    return wordpiece_ids, offsets