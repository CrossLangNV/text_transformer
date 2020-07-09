#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:53:08 2019

@author: didelani
"""
import pandas as pd
import numpy as np
import argparse
import sys
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, CharacterEmbeddings
from typing import List
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter


columns = {0: 'text', 1: 'ner'}

def train_ner(input_dir, output_dir):
    # this is the folder in which train, test and dev files reside
    data_folder = input_dir

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.tsv',
                                  test_file='test.tsv',
                                  dev_file='valid.tsv')

    print(len(corpus.train))
    print(corpus.train[1].to_tagged_string('ner'))

    # 1. get the corpus
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # 4. initialize embeddings
    embedding_types: List[TokenEmbeddings] = [

        WordEmbeddings('glove'),
        # comment in this line to use character embeddings
        #CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        # FlairEmbeddings('news-forward'),
        # FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    # 6. initialize trainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train(output_dir,
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=50)


    # 8. plot training curves (optional)
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves(output_dir+'loss.tsv')
    plotter.plot_weights(output_dir+'weights.txt')
    
    
if __name__ == '__main__':
    
    
    
    parser = argparse.ArgumentParser(description='run named entity recognition on specified dataset')
    parser.add_argument('---input_dir', type=str,
                        help='input data directory in CoNLL format')
    parser.add_argument('---output_dir', type=str,
                        help='directory where train model and test output are stored')

    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir


    if len(sys.argv) < 2:
        print('please, specify the input_dir and output_dir '
              ' \n python train_ner.py ---input_dir "../data/training/bio_cased/" ---output_dir "resources/taggers/cased-ner/" ')
        sys.exit(1)  # abort because of error

    train_ner(input_dir , output_dir)
