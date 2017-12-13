# Disamorph [![Build Status](https://travis-ci.org/peternagy1332/disamorph.svg?branch=master)](https://travis-ci.org/peternagy1332/disamorph)
## A Hungarian morphological disambiguator using sequence-to-sequence neural networks

### Usage
#### Inference
```bash
cat your_corpus | python main_inference.py -m saved_models/your_model_directory
```

#### Training
##### New model
```bash
python main_train.py -dcfg default_configs/your_config.yaml [-m path/to/save]
```

##### Continuing an existing model
```bash
python main_train.py -m saved_models/your_model_directory
```


#### Testing
##### Evaluation
```bash
python main_evaluation.py -m saved_models/your_model_directory
```

##### Accuracy estimating
```bash
python main_accuracy_estimator.py -m saved_models/your_model_directory
```

#### Commandline options
##### main_inference.py
```bash
$ python main_inference.py -h
usage: main_inference.py [-h] -m MODEL_DIRECTORY [-t]

Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence
neural networks

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
                        Path to the model directory.
  -t, --use-train-model
                        Whether to use the train insted of the validation
                        model.

```

##### main_train.py
```bash
$ python main_train.py -h
usage: main_train.py [-h] [-dcfg DEFAULT_CONFIG] [-m MODEL_DIRECTORY] [-t]

Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence
neural networks.

optional arguments:
  -h, --help            show this help message and exit
  -dcfg DEFAULT_CONFIG, --default-config DEFAULT_CONFIG
                        If provided, a new model will be trained with this
                        config. Has priority over --model-directory.
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
                        If provided, the training of an existing model will be
                        continued. If --default-config is also present, the
                        new model will be saved to this path.
  -t, --use-train-model
                        On model continuation, for defining whether to
                        continue the train insted of the validation model.
```

##### main_evaluation.py
```bash
$ python main_evaluation.py -h
usage: main_evaluation.py [-h] -m MODEL_DIRECTORY [-t]

Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence
neural networks

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
                        Path to the model directory.
  -t, --use-train-model
                        Whether to use the train instead of the validation
                        model.
```

##### main_accuracy_estimator.py
```bash
$ python main_accuracy_estimator.py -h
usage: main_accuracy_estimator.py [-h] -m MODEL_DIRECTORY [-t]

Disamorph: A Hungarian morphological disambiguator using sequence-to-sequence
neural networks

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
                        Path to the model directory.
  -t, --use-train-model
                        Whether to use the train instead of the validation
                        model.

```

#### Getting started
##### Tools
- Helsinki Finite-State Technology: https://github.com/hfst/hfst
- emMorph (Humor) Hungarian morphological analyzer: https://github.com/dlt-rilmta/emMorph

##### Installing requirements and setup as a package
```bash
pip install -r requirements.txt
python setup.py install
```

### Visualization capabilities
#### Training visualization
Character-level training
![Character-level training visualization](https://github.com/peternagy1332/disamorph/blob/master/assets/character_level_training_visualization.gif?raw=true "Character-level training visualization")

#### TensorBoard compatibility
Word embedding visualization with t-SNE. Perplexity: 5, learning rate: 10.
![Character-level word embedding visualization with t-SNE](https://github.com/peternagy1332/disamorph/blob/master/assets/character_level_tsne_per5_lr10.gif?raw=true "Character-level word embedding visualization with t-SNE")

### Training in FloydHub
``
floyd run --gpu --env tensorflow-1.3 --data peter.nagy1332/datasets/data/2:data 'ln -s /data /code/data && python main_train.py -dcfg default_configs/floydhub_morpheme_momentum_lstm.yaml -m /output'
``

### YAML configuration file structure
In these table I present the key-value pairs of the configuration YAML files with working example values.
#### Group: data
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| save_train_matrices  | false |
| train_dataset | data/szeged-judit/* |  |
| train_matrices | data/train_matrices | |
| random_seed | 448 |  |
| example_resolution | character |  |
| train_ratio | 0.8 |  |
| validation_ratio | 0.1 |  |
| batch_size | 256 |  |

#### Group: inference
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| transducer_path | /userhome/student/peterng/programs/emMorph/hfst/hu.hfstol |  |
| decoder_type | greedy |  |
| beam_width | 5 |  |

#### Group: network
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| embedding_size | 8 |  |
| hidden_layer_cell_type | LSTM |  |
| hidden_layer_cells | 64 |  |
| hidden_layer_count | 2 |  |
| max_gradient_norm | 5 |  |
| max_source_sequence_length | 109 |  |
| max_target_sequence_length | 61 |  |
| window_length | 5 |  |
| dropout_keep_probability | 0.8 |  |
| activation | null |  |

#### Group: train
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| visualization | false |  |
| epochs | 100000 |  |
| loss_optimizer | MomentumOptimizer |  |
| loss_optimizer_kwargs | {momentum: 0.5} |
| schedule | <p>- {learning_rate: 1.0,    until_global_step: 16000}<br/>- {learning_rate: 0.5,    until_global_step: 64000}<br/>- {learning_rate: 0.25,   until_global_step: 128000}<br/>- {learning_rate: 0.125,  until_global_step: 256000}<br/>- {learning_rate: 0.0625, until_global_step: 512000}<br/>- {learning_rate: 0.03125, until_global_step: 1024000}<br/>- {learning_rate: 0.015625, until_global_step: 2048000}<br/>- {learning_rate: 0.0078125, until_global_step: 4096000}<br/>- {learning_rate: 0.00390625, until_global_step: 8192000}<br/>- {learning_rate: 0.001953125, until_global_step: 16384000}<br/></p>
| shuffle_sentences | true |  |
| shuffle_examples_in_batches | false |  |
| add_summary_modulo | 100 |  |
| validation_add_summary_modulo | 100 |  |
| validation_modulo | 1 |  |


### Used shell commands during data preprocessing of Szeged Korpusz
Could be useful for train replicating purposes for different languages.

#### Collecting all analyses of words appearing in Szeged corpus
```bash
cat ./* | cut -f1 | sort | uniq | hfst-lookup --cascade=composition ../../emMorph/hfst/hu.hfstol -s | grep . | cut -f1,2 > ../analyses.txt
```

#### Non-unique words without correct analysis
```
cat * | cut -f9 | sort | grep '^$' | wc -l
```

#### Number of analyses of unique words
```
cat * | grep -e '^.' | cut -f1 | sort | uniq | hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ../../../../programs/emMorph/hfst/hu.hfstol | grep -e '.' | sort | uniq | wc -l
```

#### Number of analyses of non-unique words
```
cat * | grep -e '^.' | cut -f1 | hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ../../../../programs/emMorph/hfst/hu.hfstol | grep -e '.' | wc -l
```

### BibTex
```bibtex
@thesis{nagyp2017,
	author = {Nagy, P{\'e}ter G{\'e}za},
	title = {Hungarian morphological disambiguation using sequence-to-sequence neural networks},
	institution = {Budapest University of Technology and Economics},
	year = {2017}
}
```
