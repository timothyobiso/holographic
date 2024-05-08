# Holographic Embeddings for Text and Graphs

This repository contains the code for the paper "Holographic Embeddings for Text and Graphs" by Timothy Obiso. The paper is available on arXiv: []().

## Requirements
The code is writen in Python 3.X. Use of other versions is at your own risk! The required packages are listed in `requirements.txt`. You can install them using pip:
```
pip install -r requirements.txt
```

## Running the code
### Form Holographic Embeddings
#### Static Embeddings
This task can be run using the `run_text.py` script. The script takes the following arguments:
- the paths to the embedding sets
- `--ins` the correlation instructions separated by commas (e.g. `1,2,3`, default is `1,1,1...`)
- `--bind` the binding method to use (either `conv` or `corr`, default is `conv`)
- `--output` the name of the output file

Example:
```
python run_text.py data/word2vec.txt data/glove.txt --ins 1,2,3 --bind conv --output new_embeddings.txt
```

### Evaluate Holographic Embeddings on Text Tasks
#### Word Similarity
This task can be run using the `run_word_similarity.py` script. The script takes the following arguments:
- the paths to the embedding sets
- `--sim` the path to the word similarity dataset folder
- `--output` the name of the output file

Example:
```
python run_word_similarity.py data/word2vec.txt data/glove.txt --sim data/word_similarity --output word_similarity_results.txt
```

#### Visualize Word Embeddings
This task can be run using the `run_visualize.py` script. The script takes the following arguments:
- the paths to the embedding sets
- `--sim` the path to the word similarity dataset folder
- `--output` the name of the output file

Example:
```
python run_visualize.py data/word2vec.txt data/glove.txt --sim data/word_similarity --output word_similarity_visualization.png
```

#### MTEB Evaluation of Holographic Embeddings
This task can be run using the `run_mteb.py` script. The script takes the following arguments:
- the paths to the embeddings or the embedding model names
- - `--static` or `--contextual` to specify the type of embeddings to evaluate
  - `--static` will expect embeddings as files
  - `--contextual` will expect embeddings as models
  - If not provided, the script will default to `--static`, then try `--contextual`
- `--bind` the binding method to use (either `conv` or `corr`)

Example:
```
python run_mteb.py data/word2vec.txt data/glove.txt --bind conv --static
```

### Evaluate Holographic Embeddings on Graph Tasks
#### Triple Retrieval
This task can be run using the `run_triple.py` script. The script takes the following arguments:
- `--embedding_size` the size of the embeddings
- `--dataset` the dataset to use (default is AMR3.0)
- `--vocab` the vocabulary to use (default is AMR3.0)
- `--n` the number of examples to run (default is 1000)
- `--output` the name of the output file

Example:
```
python run_triple.py --embedding_size 1000 --dataset AMR3.0 --vocab AMR3.0 --n 1000 --output triple_e1000_1000results.txt
```

#### Graph Reformation
This task can be run using the `run_reformation.py` script. The script takes the following arguments:
- `--embedding_size` the size of the embeddings
- `--dataset` the dataset to use (default is AMR3.0)
- `--vocab` the vocabulary to use (default is AMR3.0)
- `--n` the number of examples to run (default is 1000)
- `--output` the name of the output file

Example:
```
python run_reformation.py --embedding_size 1000 --dataset AMR3.0 --vocab AMR3.0 --n 1000 --output reformation_e1000_1000results.txt
```


#### Query Vector from Question (QVQ)
This task can be run from the notebook `qvq.ipynb` or using the `run_train_qvq.py` script. The script takes the following arguments:
- `--model_size` the size of the BART model to use (default is `base`)
- `--dataset` the path to the directory containing the train and test sets

Example:
```
python run_train_qvq.py --model_size base --dataset data/qvq
```