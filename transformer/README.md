The text transformer tool helps to:
* identify sensitive words or named entities in a dialog conversation
* Text transformation of sensitive words by words of the same-type or placeholders

----

## Installation and requirements
Python 2.7 or 3 with following packages:
* flair

```
virtualen -p python3 venv
. venv/bin/activate
pip install flair
```

## Usage

```
usage: transform.py [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                    [-r {REDACT,WORD,FULL}]
                    [-m path_to_model]
                    input output
```

The default logging is INFO and the default replacement is FULL. Input file is one sentence per line. 
The default model is `ner` (automatically downloaded), but built models can be given in the command line and should be stored in the `io` directory.
