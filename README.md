The text transformer tool helps to:
* identify sensitive words or named entities in a dialog conversation
* Text transformation of sensitive words by words of the same-type or placeholders

----

## Requirements
Python 2.7 or 3 with following packages:
* numpy
* flair

## Usage
### Preprocess named entity recognition dataset. 
Given an annotated text file(s) in a CoNLL format, create 4 kinds of datasets that will be stored in the directory "data/training/" i.e training data with : 
* cased words and punctuations
* uncased words and  punctuations
* cased words but without punctuations
* uncased words but without punctuations

> python preprocess.py [data_path_conll_format]


### Create Text Transformation dataset by replacing sensitive words with placeholders or words of the same entity type.
Get the list of all available words in each entity type and sort them by relative frequency of their occurence in the corpus and

Transform text according to the following replacement strategy:
* Placeholder replacement - every occurence of a named entity word is replaced by a "Placeholder" token
* Word-by-word / singleword sametype replacement  - every named entity word is replaced by another word of the same type without paying attention to multiword expressions
* Full entity / multiword sametype replacement - multiword expression is replaced by another multi-word expression and single word expression replaced by another single word expression

> python create_privacy_transformed_data.py


The named entity recognition (NER) is trained using [Flair] (https://github.com/zalandoresearch/flair) that is based on BiLSTM-CRF model and word features are obtained from Glove embeddings. 

To train the NER model, use:
> python train_ner.py ---input_dir training_data_dir ---output_dir model_output_dir

`training_data_dir`  the directory consisting of the training, development and test data in '.tsv' extension

`model_output_dir`  the directory where the model output, and training log is stored
 


### Demo of text transformation, it accepts a sentence from the command line
* automatically identify the sensitive words, and 
* replace each single word labeled as named entity using either placeholder, Word-by-word or full entity replacement strategy

> python demo_text_transformation.py


> Enter "exit" to quit or sample sentence ' 

> Enter a sample sentence: Mark Smith is going to London

> Tagged sentence:  Mark &lt; B-PER> Smith &lt; I-PER> is going to London &lt; B-LOC>


> Enter "exit" to quit 

> Enter 0 for Redact NE replacement or  Enter 1 for Word-by-Word replacement or  Enter 2 for Full-Entity replacement: 0 or Enter "exit" to quit 

> Placeholder selected:  0

> 0 Transformed sentence:  '▮▮▮▮▮ ▮▮▮▮▮'<PER> is going to '▮▮▮▮▮'<LOC> 


> Enter 0 for Redact NE replacement or  Enter 1 for Word-by-Word replacement or Enter 2 for Full-Entity replacement: 1 Enter "exit" to quit 

> Word-by-word selected:  1

> 1 Transformed sentence:  'Tijjani Boer' &lt;PER> is going to 'Jordan' &lt;LOC> 



> Enter 0 for Redact NE replacement or  Enter 1 for Word-by-Word replacement or Enter 2 for Full-Entity replacement: 1 Enter "exit" to quit 


> Full-entity selected:  2

> 2 Transformed sentence:  'Monica Seles' &lt;PER> is going to 'Britain'&lt;LOC> 
