#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:31:38 2019

@author: didelani
"""
import os
import sys
import numpy as np
from collections import defaultdict, Counter
import re
import json


output_dir = '../data/training/'

def get_tagged_sentence(sent):
    tagged_sentence = ''
    for token, ne in sent:
        tagged_sentence += token
        if ne!='O':
            tagged_sentence += '<'+ne+'>'+ ' ' 
        else:
            tagged_sentence += ' ' 
    return tagged_sentence


def extract_sentences(data_dir):
    data_path = data_dir + 'train.tsv'
    with open(data_path) as f:
        docs = f.readlines()

    sentences = []
    n_sent = 0
    sent = []
    list_sent_len = []

    named_entities = defaultdict(lambda: defaultdict(int))
    multi_word_entites = defaultdict(lambda: defaultdict(int))
    sent_per = 0
    print('# tokens', len(docs))
    for i, line in enumerate(docs):
        if len(line) < 3:
            if len(sent) > 0:
                sentences.append(sent)
            sent = []
            list_sent_len.append(sent_per)
            sent_per = 0
            n_sent += 1
        else:
            token, ne = line.strip().split('\t')
            sent.append((token, ne))
            named_entities[ne][token] += 1

            k = i
            new_token = token + ' '
            while k + 1 < len(docs) and len(docs[k + 1]) > 3 and ne != 'O' and ne[:2] == 'B-':

                n_tok, n_ne = docs[k + 1].strip().split('\t')

                if ne[2:] == n_ne[2:] and n_ne[:2] == 'I-':
                    new_token += n_tok + ' '
                    k += 1
                else:
                    break

            sent_per += 1

            if len(new_token) > len(token) + 1:
                named_entities['MULTI-WORD_' + ne[2:]][new_token] += 1
    print('# of sentences ', len(sentences))

    #for i in range(len(sentences[:10])):
    #    print(get_tagged_sentence(sentences[i]))

    print('# multi-word entity categories', len(multi_word_entites))

    return sentences, list_sent_len, named_entities, multi_word_entites


def write_ne_dict_To_file(text_data, output_dir):
    with open(output_dir, 'w') as fwriter:
        for token_cnt_prob in text_data:
            token, cnt, prob = token_cnt_prob
            fwriter.write(token+'\t'+str(cnt)+ '\t'+str(prob)+'\n')

def write_tsv_file(text_data, output_dir):
    with open(output_dir, 'w') as fwriter:
        for token_tag in text_data:
            token, tag = token_tag
            fwriter.write(token+'\t'+tag+'\n')

def read_conll_data(data_dir_file):
    with open(data_dir_file) as f:
        text_lines = f.readlines()

    tokens_tags = []
    for line in text_lines:
        token_tag = line.strip().split('\t')
        if len(token_tag) > 1:
            token, tag =  token_tag
        else:
            token, tag = '', ''

        tokens_tags.append([token, tag])

    return tokens_tags



def sort_named_entities_by_freq(transf_dir, named_entities):
    
    named_entities_freq = []

    named_entity_to_idx = dict()
    
    k = 0
    for named_entity in named_entities:
        ne_dict = Counter(named_entities[named_entity])
        total_sum = sum(ne_dict.values())
    
        # compute the probability of ne in corpus
        ne_counts = sorted(ne_dict.items(), key=lambda x: x[1], reverse=True)

        nes = []
        for ne, cnt in ne_counts:
            nes.append([ne, cnt, cnt/total_sum])

        named_entities_freq.append(nes)
        named_entity_to_idx[named_entity]=k
        k+=1
        print(named_entity, nes[:5])

        write_ne_dict_To_file(nes, transf_dir+named_entity+'.tsv')

    with open(transf_dir+'named_entity_to_idx.json', 'w') as f:
        json.dump(named_entity_to_idx, f)
        
    return named_entities_freq, named_entity_to_idx


def check_multiword(sent_o, tag_prefix = 'singleword'):
    sent_new = []
    n = len(sent_o)
    i=0
    while i < n:
        word_feat = sent_o[i]
        if word_feat[1] != 'O':
            j=i
            act_label = word_feat[1][2:]
            n_first_ne = word_feat[0]
            word_feat = (n_first_ne, word_feat[1])
            while j+1 < n and sent_o[j+1][1][2:] == act_label:
                tag =  re.split('[_ -]', act_label)[-1]
                if tag_prefix == 'multiword':
                    tag = 'MULTI-WORD_'+tag
                else:
                    tag = 'B-'+tag
                word_feat = (n_first_ne, tag)
                j+=1
            i=j
            # change single-word expression to have multiword NE type
            '''
            if tag_prefix == 'multiword':
                tag =  re.split('[_ -]', act_label)[-1]
                tag = 'MULTI-WORD_'+tag
                word_feat = (n_first_ne, tag)
            '''
        sent_new.append(word_feat)
        i+=1
    return sent_new


def anonymize_corpus_placeholder(sentences, placeholderType, multi_word=True):
    sentences_ph = []
    new_sentences_ph = []
    for j, sent in enumerate(sentences):
        sent_ph = []
        new_sent = []

        if multi_word:
            sent = check_multiword(sent)

        for k, word_label in enumerate(sent):
            token, ne = word_label

            new_token = token
            if ne != 'O':
                new_token = 'Placeholder'
            else:
                new_token = token

            new_sent.append((new_token, ne))
            sent_ph.append((token, ne))

        sentences_ph.append(sent_ph)
        new_sentences_ph.append(new_sent)

    return new_sentences_ph


def anonymize_corpus_same_type(list_sent_len, sentences, ne_table, ne_label='B-TIME', multi_word=False):
    N_words = np.max(list_sent_len)
    N_sents = len(sentences)

    N_popNE = len(ne_table)
    ne_prob = list(zip(*ne_table))[-1]
    ne_list = list(zip(*ne_table))[0]
    sel_ne_ids = np.random.choice(range(N_popNE), (N_sents, N_words), p=ne_prob)

    new_sentences = []
    anony_ne_map = defaultdict(list)
    for j, sent in enumerate(sentences):
        new_sent = []
        # print(sent)
        per_NE = dict()
        per_no = 0

        if multi_word:
            sent = check_multiword(sent)

        for k, word_label in enumerate(sent):
            token, ne = word_label
            N_words = len(sent)
            new_token = token
            if ne == ne_label:
                if token not in per_NE:
                    ne_idx = sel_ne_ids[j][per_no]
                    new_token = ne_list[ne_idx]
                    per_no += 1
                    per_NE[token] = new_token
                else:
                    new_token = per_NE[token]
                # print(token, new_token)
                anony_ne_map[new_token].append((token, j, k))

            new_sent.append((new_token, ne))

        new_sentences.append(new_sent)

    # print an example of anonymized sentence

    return new_sentences


# multiword to multiword text transformation
def anonymize_same_type_multiword(list_sent_len, sentences, ne_table, ne_label, multi_word=False):
    N_words = np.max(list_sent_len)
    N_sents = len(sentences)

    N_popNE = len(ne_table)
    ne_prob = list(zip(*ne_table))[-1]
    ne_list = list(zip(*ne_table))[0]
    sel_ne_ids = np.random.choice(range(N_popNE), (N_sents, N_words), p=ne_prob)

    new_sentences = []
    anony_ne_map = defaultdict(list)
    for j, sent in enumerate(sentences):
        new_sent = []
        per_NE = dict()
        per_no = 0

        for k, word_label in enumerate(sent):
            token, ne = word_label
            N_words = len(sent)
            new_token = token
            if ne == ne_label:
                if token not in per_NE:
                    ne_idx = sel_ne_ids[j][per_no]
                    new_token = ne_list[ne_idx]
                    per_no += 1
                    per_NE[token] = new_token
                else:
                    new_token = per_NE[token]
                # print(token, new_token)
                anony_ne_map[new_token].append((token, j, k))

            new_sent.append((new_token, ne))

        new_sentences.append(new_sent)

    return new_sentences


def tranform_text_and_save(sentences, list_sent_len, sorted_ne_freq, ne_to_idx, replacementType="Placeholder",
                           multi_word=False):
    folder_name = ''
    if replacementType == "Placeholder":
        anony_sentences = anonymize_corpus_placeholder(sentences, replacementType)
        folder_name = "placeholder"
        print('Placeholder transformed text: \n', get_tagged_sentence(anony_sentences[5]))
    elif replacementType == "SametypeSingleword":
        folder_name = "sametype_singleword"
        anony_sentences = sentences
        for ne in ne_to_idx:
            ne_freq = sorted_ne_freq[ne_to_idx[ne]]
            if 'MULTI-WORD' not in ne and ne != 'O':
                anony_sentences = anonymize_corpus_same_type(list_sent_len, anony_sentences, ne_freq, ne)
    else:
        folder_name = "sametype_multiword"

        anony_sentences = []
        for j, sent in enumerate(sentences):
            sent = check_multiword(sent, tag_prefix='multiword')
            anony_sentences.append(sent)

        print('Text sample sentence: \n', get_tagged_sentence(sentences[5]))
        print('Multiword text identified: \n', get_tagged_sentence(anony_sentences[5]))
        # Step 1: multi-word expressions replacement
        for ne in ne_to_idx:
            ne_freq = sorted_ne_freq[ne_to_idx[ne]]
            if 'MULTI-WORD' in ne and ne != 'O':
                anony_sentences = anonymize_same_type_multiword(list_sent_len, anony_sentences, ne_freq, ne)

        # Step 2: singleword replacement
        for ne in ne_to_idx:
            ne_freq = sorted_ne_freq[ne_to_idx[ne]]
            if 'MULTI-WORD' not in ne and ne != 'O':
                anony_sentences = anonymize_corpus_same_type(list_sent_len, anony_sentences, ne_freq, ne)
        print('MULTI-WORD sentence transformation: \n', get_tagged_sentence(anony_sentences[5]))

    if replacementType == "SametypeMultiword":
        anonynimized_sentence = []
        for sent in anony_sentences:
            for token, tag in sent:
                multi_words = token.split()
                new_tag = re.split('[_ -]', tag)[-1]
                if 'MULTI-WORD' in tag:
                    anonynimized_sentence.append([multi_words[0], 'B-' + new_tag])
                    for word in multi_words[1:]:
                        anonynimized_sentence.append([word, 'I-' + new_tag])
                else:
                    anonynimized_sentence.append([token, tag])
            anonynimized_sentence.append(['', ''])
    else:
        anonynimized_sentence = []
        for sent in anony_sentences:
            for token, tag in sent:
                anonynimized_sentence.append([token, tag])
            anonynimized_sentence.append(['', ''])

    new_output_dir = output_dir + folder_name + '/'
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    write_tsv_file(anonynimized_sentence, new_output_dir+ 'train.tsv')
    valid_sentences =  read_conll_data(data_dir + 'valid.tsv')
    write_tsv_file(valid_sentences, new_output_dir + 'valid.tsv')
    test_sentences = read_conll_data(data_dir + 'test.tsv')
    write_tsv_file(test_sentences, new_output_dir + 'test.tsv')



if __name__ == '__main__':
    data_dir = '../data/training/bio_cased/'

    if len(sys.argv) < 2:
        print("use the default data path: '../data/training/bio_cased/' ")
    else:
        print("Please run the code with: python create_privacy_transformed_data.py [data_path_conll_format] e.g")
        print("python create_privacy_transformed_data.py '../data/training/bio_cased/'  ")
        data_dir = sys.argv[1]

    sentences, list_of_sentence_len, named_entities, multi_word_entites = extract_sentences(data_dir)
    #print(sentences[8816])
    transf_dir = '../data/transformation/'
    if not os.path.exists(transf_dir):
        os.makedirs(transf_dir)

    sorted_ne_freq, ne_to_idx = sort_named_entities_by_freq(transf_dir, named_entities)

    tranform_text_and_save(sentences, list_of_sentence_len, sorted_ne_freq, ne_to_idx, replacementType="Placeholder",
                           multi_word=True)
    tranform_text_and_save(sentences, list_of_sentence_len, sorted_ne_freq, ne_to_idx,
                           replacementType="SametypeSingleword")
    tranform_text_and_save(sentences, list_of_sentence_len, sorted_ne_freq, ne_to_idx,
                           replacementType="SametypeMultiword", multi_word=True)
