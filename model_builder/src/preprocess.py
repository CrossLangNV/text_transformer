import os
import sys
import string
import numpy as np

# create 4 types of datasets:
# * cased words with punctuations
# * uncased words with punctuations
# * cased words without punctuations
# * uncased words without punctuations

# get token and tag for a text file in CoNLL format
punctuations = set(string.punctuation)
def preprocess(input_data, isLower=False, no_punctuations=False):
    with open(input_data) as f:
        text_lines = f.readlines()

    tokens_tags = []
    # ignore the first 2 lines in the CONLL data
    for line in text_lines[2:]:
        per_token_attr = line.strip().split(" ")
        if len(per_token_attr)> 1:
            token = per_token_attr[0]
            tag = per_token_attr[-1]
        else:
            token = ''
            tag = ''

        if isLower:
            token = token.lower()

        if no_punctuations:
            if token not in punctuations:
                tokens_tags.append([token, tag])
        else:
            tokens_tags.append([token, tag])

    return tokens_tags

def write_tsv_file(text_data, output_dir):
    with open(output_dir, 'w') as fwriter:
        for token_tag in text_data:
            token, tag = token_tag
            fwriter.write(token+'\t'+tag+'\n')


def create_data(input_data, output_dir, isLower=False, no_punctuations=False):
    train_data = preprocess(input_data+'/train.txt', isLower, no_punctuations)
    valid_data = preprocess(input_data+'/valid.txt', isLower, no_punctuations)
    test_data = preprocess(input_data+'/test.txt', isLower, no_punctuations)

    write_tsv_file(train_data, output_dir+'/train.tsv')
    write_tsv_file(valid_data, output_dir+'/valid.tsv')
    write_tsv_file(test_data, output_dir+'/test.tsv')


if __name__ == '__main__':

    input_dir  = '../data/conll2003/'
    if len(sys.argv) < 2:
        print("use the default data path: ../data/conll2003/")
    else:
        print("Please run the code with: python preprocess.py [data_path_conll_format] e.g")
        print("python preprocess.py '../data/conll2003/' ")
        input_data_path = sys.argv[1]

    output_dir = '../data/training/bio_uncased/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_data(input_dir, output_dir, isLower = True)
    print('training data with uncased words with punctuations in: ',output_dir)

    output_dir = '../data/training/bio_cased/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_data(input_dir, output_dir, isLower = False)
    print('training data with cased words with punctuations in: ',output_dir)

    
    output_dir = '../data/training/bio_uncased_nopunct/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_data(input_dir, output_dir, isLower = True, no_punctuations=True)
    print('training data with uncased words without punctuations in: ',output_dir)


    output_dir = '../data/training/bio_cased_nopunct/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_data(input_dir, output_dir, isLower = False, no_punctuations=True)
    print('training data with cased words without punctuations in: ',output_dir)

    
