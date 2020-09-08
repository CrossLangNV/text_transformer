#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:59:22 2019

@author: didelani
"""
import re
import numpy as np
import json
from flair.data import Sentence
from flair.models import SequenceTagger
import logging

logger = logging.getLogger(__name__)

# model_path = 'model/best-model.pt'
transf_dir = 'data/transformation/'


def read_ne_dict(data_dir_file):
    with open(data_dir_file) as f:
        text_lines = f.readlines()

    token_cnt_probs = []
    for line in text_lines:
        tok_cnt_prob = line.strip().split('\t')
        token, count, prob = tok_cnt_prob
        token_cnt_probs.append([token, count, prob])

    return token_cnt_probs


def get_tagged_sentence(sent):
    tagged_sentence = ''
    for token, ne in sent:
        tagged_sentence += token
        if ne != 'O':
            tagged_sentence += '<'+ne+'>' + ' '
        else:
            tagged_sentence += ' '
    return tagged_sentence


def take_to_bio_format(labels):
    bio_labels = []
    for k, label in enumerate(labels):
        ne = label
        if k > 0 and len(labels[k - 1]) > 2:
            prev_ne = labels[k - 1]
        else:
            prev_ne = 'O'
        new_ne = ne
        if new_ne != 'O':
            ne = 'B-' + new_ne
            if new_ne == prev_ne:
                ne = 'I-' + new_ne

        bio_labels.append(ne)

    return bio_labels

def identify_private_tokens_in_sentences(sents, model):
    sentences = [Sentence(sent.strip()) for sent in sents]
    # predict tags and print
    model.predict(sentences)
    tagged_sentences = [sent.to_tagged_string() for sent in sentences]

    all_tagged_sents, all_coNLL_format_tags  = [], []

    for tagged_sent in tagged_sentences:
        tagged_words = tagged_sent.split()
        coNLL_format_tags = []

        for k, tagged_word in enumerate(tagged_words):
            token = tagged_word
            if token in ["<unk>", "[noise1]", "[noise2]", "umm", "ahh"]:
                tag = 'O'
                coNLL_format_tags.append([token, tag])
                continue
            if token.startswith('<') and token.endswith('>'):
                continue

            if k < len(tagged_words) - 1:
                next_token = tagged_words[k + 1]
            else:
                next_token = ''

            if next_token.startswith('<') and next_token.endswith('>'):
                tag = next_token[1:-1]
            else:
                tag = 'O'
            if tag != 'O':
                coNLL_format_tags.append([token, tag[2:]])
            else:
                coNLL_format_tags.append([token, tag])

        unzip = list(zip(*coNLL_format_tags))
        tags = take_to_bio_format(list(unzip[1]))
        coNLL_format_tags = list(zip(list(unzip[0]), tags))

        all_tagged_sents.append(tagged_sent)
        all_coNLL_format_tags.append(coNLL_format_tags)

    return all_tagged_sents, all_coNLL_format_tags


def identify_private_tokens(sentr, model):

    sentence = Sentence(sentr.strip())

    # predict tags and print
    model.predict(sentence)
    tagged_sent = sentence.to_tagged_string()

    coNLL_format_tags = []

    tagged_words = tagged_sent.split()
    for k, tagged_word in enumerate(tagged_words):
        token = tagged_word
        if token.startswith('<') and token.endswith('>'):
            continue

        if k < len(tagged_words) - 1:
            next_token = tagged_words[k + 1]
        else:
            next_token = ''

        if next_token.startswith('<') and next_token.endswith('>'):
            tag = next_token[1:-1]
        else:
            tag = 'O'
        if tag != 'O':
            coNLL_format_tags.append([token, tag[2:]])
        else:
            coNLL_format_tags.append([token, tag])

    unzip = list(zip(*coNLL_format_tags))
    tags = take_to_bio_format(list(unzip[1]))
    coNLL_format_tags = list(zip(list(unzip[0]), tags))

    return tagged_sent, coNLL_format_tags


def getTaggedString(per_sent_tokens, predicted_tags):
    token_idxs = np.arange(len(per_sent_tokens))
    tagged_string = ""
    t = 0
    while t < len(token_idxs):
        token, tag = per_sent_tokens[t], predicted_tags[t]

        k = t
        new_token = token + ' '
        while (k + 1 < len(per_sent_tokens) and
               len(per_sent_tokens[k + 1]) > 0 and
               tag != 'O' and tag[:2] == 'B-'):

            n_tok, n_ne = per_sent_tokens[k + 1], predicted_tags[k + 1]
            if tag[2:] == n_ne[2:] and n_ne[:2] == 'I-':
                new_token += n_tok + ' '
                k += 1
            else:
                break
        t = k
        t += 1

        if tag != "O":
            tagged_string += new_token[:-1] + " "
            #tagged_string += "'" + new_token[:-1]+"'<"+tag[2:]+"> "
        else:
            tagged_string += token + " "


    return tagged_string


def check_multiword(sent_o, tag_prefix='singleword'):
    sent_new = []
    n = len(sent_o)
    i = 0
    while i < n:
        word_feat = sent_o[i]
        if word_feat[1] != 'O':
            j = i
            act_label = word_feat[1][2:]
            n_first_ne = word_feat[0]
            word_feat = (n_first_ne, word_feat[1])
            while j+1 < n and sent_o[j+1][1][2:] == act_label:
                tag = re.split('[_ -]', act_label)[-1]
                if tag_prefix == 'multiword':
                    tag = 'MULTI-WORD_'+tag
                else:
                    tag = 'B-'+tag
                word_feat = (n_first_ne, tag)
                j += 1
            i = j
            # change single-word expression to have multiword NE type
            '''
            if tag_prefix == 'multiword':
                tag =  re.split('[_ -]', act_label)[-1]
                tag = 'MULTI-WORD_'+tag
                word_feat = (n_first_ne, tag)
            '''
        sent_new.append(word_feat)
        i += 1
    return sent_new


def anonymize_sentence_singleword(sent, ne_table_list, nes_to_idxs):
    N_words = 50

    for ne_label, idx in nes_to_idxs.items():
        if ne_label == 'O':
            continue
        ne_table = ne_table_list[idx]

        N_popNE = len(ne_table)
        ne_prob = list(zip(*ne_table))[-1]
        ne_list = list(zip(*ne_table))[0]
        sel_ne_ids = np.random.choice(range(N_popNE), (1, N_words), p=ne_prob)
        sel_ne_ids = sel_ne_ids.flatten()

        new_sent = []
        # print(sent)
        per_NE = dict()
        per_no = 0
        for k, word_label in enumerate(sent):
            token, ne = word_label
            N_words = len(sent)
            new_token = token
            if ne == ne_label:
                if token not in per_NE:
                    ne_idx = sel_ne_ids[per_no]
                    new_token = ne_list[ne_idx]
                    per_no += 1
                    per_NE[token] = new_token
                else:
                    new_token = per_NE[token]
                # print(token, new_token)
            new_sent.append((new_token, ne))

        sent = new_sent

    new_sent = sent

    return new_sent


# multiword to multiword text transformation
def anonymize_sentence_multiword(sent, ne_table_list, nes_to_idxs):
    sent = check_multiword(sent, tag_prefix='multiword')

    N_words = 50

    for ne_label, idx in nes_to_idxs.items():
        if ne_label == 'O' or 'MULTI-WORD' not in ne_label:
            continue
        ne_table = ne_table_list[idx]

        N_popNE = len(ne_table)
        ne_prob = list(zip(*ne_table))[-1]
        ne_list = list(zip(*ne_table))[0]
        sel_ne_ids = np.random.choice(range(N_popNE), (1, N_words), p=ne_prob)
        sel_ne_ids = sel_ne_ids.flatten()

        new_sent = []

        # if 'MULTI-WORD' in ne_label:
        #    sent = check_multiword(sent, tag_prefix='multiword')

        per_NE = dict()
        per_no = 0
        for k, word_label in enumerate(sent):
            token, ne = word_label
            N_words = len(sent)
            new_token = token
            if ne == ne_label:
                if token not in per_NE:
                    ne_idx = sel_ne_ids[per_no]
                    new_token = ne_list[ne_idx]
                    per_no += 1
                    per_NE[token] = new_token
                else:
                    new_token = per_NE[token]
                # print(token, new_token)
            new_sent.append((new_token, ne))

        sent = new_sent

    new_sent = anonymize_sentence_singleword(sent, ne_table_list, nes_to_idxs)

    anonynimized_sentence = []
    for token, tag in new_sent:
        multi_words = token.split()
        new_tag = re.split('[_ -]', tag)[-1]
        if 'MULTI-WORD' in tag:
            anonynimized_sentence.append([multi_words[0], 'B-' + new_tag])
            for word in multi_words[1:]:
                anonynimized_sentence.append([word, 'I-' + new_tag])
        else:
            anonynimized_sentence.append([token, tag])
    return anonynimized_sentence


def anonymize_corpus_placeholder(sent):
    sent_ph = []
    new_sent = []

    for k, word_label in enumerate(sent):
        token, ne = word_label
        new_token = token
        if ne != 'O':
            # new_token = 'PLACEHOLDER'
            new_token = '▮▮▮▮▮'

        new_sent.append((new_token, ne))
        sent_ph.append((token, ne))

    return new_sent


def transform_private_tokens(sentence, ne_table_list, nes_to_idxs,  i=0):
    if i == 'REDACT':
        logger.info("Placeholder selected: {i}")
        new_sent = anonymize_corpus_placeholder(sentence)
    elif i == 'WORD':
        logger.info("Word-by-word selected: {i}")
        new_sent = anonymize_sentence_singleword(sentence,
                                                 ne_table_list,
                                                 nes_to_idxs)
    elif i == 'FULL':
        logger.info("Full-entity selected: {i}")
        new_sent = anonymize_sentence_multiword(sentence,
                                                ne_table_list,
                                                nes_to_idxs)
    else:
        pass

    tokens, tags = list(zip(*new_sent))

    tagged_string = getTaggedString(list(tokens), list(tags))

    return tagged_string


def get_named_entities():

    with open(transf_dir + 'named_entity_to_idx.json') as f:
        nes_to_idxs = json.load(f)

    ne_table_list = ['' for _ in range(len(nes_to_idxs))]
    for ne, idx in nes_to_idxs.items():
        ne_table_list[idx] = read_ne_dict(transf_dir + ne + '.tsv')

    return nes_to_idxs, ne_table_list


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Transform texts removing sensitive words, named entities')
    parser.add_argument("-l", "--log",
                        dest="logLevel",
                        choices=['DEBUG', 'INFO',
                                 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help="Set the logging level",
                        default='INFO')
    parser.add_argument('-r',
                        help='sets the replace type',
                        dest='replace_type',
                        choices=['REDACT', 'WORD', 'FULL'],
                        default='FULL')
    parser.add_argument('-m',
                        help='model for named entity recognition',
                        dest='model')
    parser.add_argument('-data_format',
                        help='sets the data preprocessing format',
                        dest='data_format',
                        choices=['cased', 'uncased', 'cased_nopunct', 'uncased_nopunct'],
                        default='cased')
    parser.add_argument('input',
                        help="file of sentences",
                        type=argparse.FileType('r'))
    parser.add_argument('output',
                        help="result file path",
                        type=argparse.FileType('w'))

    args = parser.parse_args()

    if args.model:
        model = args.model
    else:
        model = 'ner'

    transf_dir = transf_dir + 'bio_'+args.data_format+'/'

    logging.basicConfig(filename='text_transformer.log',
                        filemode='w',
                        level=getattr(logging,
                                      args.logLevel))

    # Get named entities
    nes_to_idxs, ne_table_list = get_named_entities()

    logger.info("File : {}\n".format(args.input))

    model = SequenceTagger.load(model)

    '''
    for sent in args.input:

        logger.info("Sentence: {}\n".format(sent))

        tagged_sent, tokens_tags = identify_private_tokens(sent, model)
        logger.info("Tagged sentence: {}\n".format(tagged_sent))

        tagged_string = transform_private_tokens(tokens_tags,
                                                 ne_table_list,
                                                 nes_to_idxs,
                                                 i=args.replace_type)
        args.output.write(f"{tagged_string}\n")
    '''

    all_sent = []
    for sent in args.input:
        all_sent.append(sent)

    all_tagged_sents, all_sents_tokens_tags = identify_private_tokens_in_sentences(all_sent, model)
    for i in range(len(all_tagged_sents)):
        tagged_sent = all_tagged_sents[i]
        tokens_tags = all_sents_tokens_tags[i]
        logger.info("Tagged sentence: {}\n".format(tagged_sent))

        tagged_string = transform_private_tokens(tokens_tags,
                                                 ne_table_list,
                                                 nes_to_idxs,
                                                 i=args.replace_type)
        args.output.write(f"{tagged_string}\n")


    args.output.close()
