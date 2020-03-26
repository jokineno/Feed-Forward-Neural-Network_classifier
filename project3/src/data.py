import os
from collections import defaultdict as dd
import nltk
import torch

WORD_BOUNDARY='#'
UNK='UNK'
HISTORY_SIZE=4

def get_examples(tok_word,history_size, character_map):
    ngrams = nltk.ngrams(tok_word,
                         history_size+1,
                         pad_left=1,
                         left_pad_symbol=WORD_BOUNDARY,
                         pad_right=1,
                         right_pad_symbol=WORD_BOUNDARY)    
    ngrams = [[character_map[c] if c in character_map else character_map[UNK] 
               for c in ngram]
              for ngram in ngrams]
    
    return [(torch.LongTensor(ngram[:-1]),
             torch.LongTensor([ngram[-1]])) for ngram in ngrams]

def read_words(fn):
    data = []
    for line in open(fn, encoding='utf-8'):
        line = line.strip('\n')
        if line:
            wf, lan = line.split('\t')
            data.append({'WORD':wf, 
                         'TOKENIZED WORD':[c for c in wf], 
                         'LANGUAGE':lan})
    return data 

def compute_character_tuples(word_ex,character_map):
    word_ex['TUPLES'] = get_examples([c for c in word_ex['TOKENIZED WORD']],
                                     HISTORY_SIZE,
                                     character_map)


def sort_by_language(dataset):
    sorteddata = dd(lambda : [])
    for ex in dataset:
        sorteddata[ex['LANGUAGE']].append(ex)
    return sorteddata

def read_datasets(prefix,data_dir):
    datasets = {'training': read_words(os.path.join(data_dir, '%s.%s' % 
                                                    (prefix, 'train'))), 
                'dev': read_words(os.path.join(data_dir, '%s.%s' % 
                                               (prefix, 'dev'))),
                'test': read_words(os.path.join(data_dir, '%s.%s' %
                                                (prefix, 'test')))} 

    charmap = {c:i for i,c in enumerate({c for ex in datasets['training'] 
                                         for c in ex['TOKENIZED WORD']})}
    charmap[UNK] = len(charmap)
    charmap[WORD_BOUNDARY] = len(charmap)
    languages = {ex['LANGUAGE'] for ex in datasets['training']}

    for word_ex in datasets['training']:
        compute_character_tuples(word_ex,charmap)
    for word_ex in datasets['dev']:
        compute_character_tuples(word_ex,charmap)
    for word_ex in datasets['test']:
        compute_character_tuples(word_ex,charmap)

    datasets['training'] = sort_by_language(datasets['training'])

    return datasets, charmap, languages

if __name__=='__main__':
    from paths import data_dir
    d,c = read_datasets('uralic',data_dir)


