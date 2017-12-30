# Disamorph [![Build Status](https://travis-ci.org/peternagy1332/disamorph.svg?branch=master)](https://travis-ci.org/peternagy1332/disamorph)
## A Hungarian morphological disambiguator using sequence-to-sequence neural networks

### Usage
#### Inference
```bash
cat your_corpus | python main_inference.py -m saved_models/your_model_directory [-t]
```
##### Example
```bash
$ echo "Próbáld meg még egyszer." | python main_inference.py -m saved_models/LSTMx4x64.Momentum.tanh.morp
Próbáld     próbál[/V]d[Sbjv.Def.2Sg]
meg     meg[/Prev]
még     még[/Adv]
egyszer     egy[/Num]szer[_Mlt-Iter/Adv]
.   .
```
Note that the delimiter is a tab (`\t`) character in each line.

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
neural networks.

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
neural networks.

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
neural networks.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIRECTORY, --model-directory MODEL_DIRECTORY
                        Path to the model directory.
  -t, --use-train-model
                        Whether to use the train instead of the validation
                        model.

```

#### Getting started
1. Please install all the following tools:
    - Helsinki Finite-State Technology: https://github.com/hfst/hfst
    - emMorph (Humor) Hungarian morphological analyser: https://github.com/dlt-rilmta/emMorph
2. Make sure that `transducer_path` is set correctly in your `model_configuration.yaml` in your model's directory.
3. Install the Python requirements: `pip install -r requirements.txt`
4. Optionally, if you wish to use the Python API of this project, install it as a package: `python setup.py install`

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
| save_train_matrices  | true | Cache preprocessed corpus matrices to the filesystem. Default: true
| train_dataset | data/szeged-judit/* | Szeged Corpus. Please request these files on demand.
| train_matrices | data/train_matrices | Where to save train matrices.
| random_seed | 448 |  To make random number generation deterministic and experiments reproducible.
| example_resolution | character | Is it a `character` or `morpheme`-level model?
| train_ratio | 0.8 |  First 80% of sentences are used for training.
| validation_ratio | 0.1 |  First 10% of sentences after the first 80% is for validation. The remaining is the test dataset.
| batch_size | 256 |  For all models.

#### Group: inference
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| transducer_path | /userhome/student/peterng/programs/emMorph/hfst/hu.hfstol |  See requirements.
| decoder_type | greedy |  Or `beam_decoder`.
| beam_width | 5 |  Only needed when `decoder_type` is `beam_decoder`.

#### Group: network
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| embedding_size | 8 |  Word embedding vector lengths.
| hidden_layer_cell_type | LSTM |  Or `GRU`. Used for both encoder and decoder networks in all models.
| hidden_layer_cells | 64 |  How many cells are in a layer?
| hidden_layer_count | 2 |  Should be an even number because of the bidirectional encoder.
| max_gradient_norm | 5 |  Maximum gradient clipping.
| max_source_sequence_length | 109 | The maximum real sequence length in all train matrices. See the data preprocessor class to find this out.
| max_target_sequence_length | 61 |  Same as `max_source_sequence_length`.
| window_length | 5 |  Sliding window size that moves one-word right from the beginning of each sentence.
| dropout_keep_probability | 0.8 |  Note that this is not in inverse-notation!
| activation | null |  Anything on `tf.nn`. E.g. `tanh`, `sigmoid`, `relu`, `leaky_relu`.

#### Group: train
| Key                                    | Example value            | Explanation                           |
-----------------------------------------|--------------------------|---------------------------------------|
| visualization | false |  If true, train visualisation is turned on, as seen in the first GIF. I used this only for test purposes with `data_batch_size=1`
| epochs | 100000 |  # of training epochs.
| loss_optimizer | MomentumOptimizer |  Anything on `tf.train`. E.g. `AdamOptimizer`, `RMSPropOptimizer`, ...
| loss_optimizer_kwargs | {momentum: 0.5} | Additional kwargs for the optimizer if needed. Default: {}
| schedule | <p>- {learning_rate: 1.0,    until_global_step: 16000}<br/>- {learning_rate: 0.5,    until_global_step: 64000}<br/>- {learning_rate: 0.25,   until_global_step: 128000}<br/>- {learning_rate: 0.125,  until_global_step: 256000}<br/>- {learning_rate: 0.0625, until_global_step: 512000}<br/>- {learning_rate: 0.03125, until_global_step: 1024000}<br/>- {learning_rate: 0.015625, until_global_step: 2048000}<br/>- {learning_rate: 0.0078125, until_global_step: 4096000}<br/>- {learning_rate: 0.00390625, until_global_step: 8192000}<br/>- {learning_rate: 0.001953125, until_global_step: 16384000}<br/></p> | Decaying learning rate. See `stats/schedule_generator.py`.
| shuffle_sentences | true |  Only train dataset.
| shuffle_examples_in_batches | false | Does not really make sense, since the order of examples matters.
| add_summary_modulo | 100 |  Log at every 100th training step.
| validation_add_summary_modulo | 100 |  Log every 100th validation step.
| validation_modulo | 1 |  Validate after every epoch.


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

### Similar projects
- https://github.com/nagyniki017/morph_disamb

### BibTex
```bibtex
@thesis{nagyp2017,
	author = {Nagy, P{\'e}ter G{\'e}za},
	title = {Hungarian morphological disambiguation using sequence-to-sequence neural networks},
	institution = {Budapest University of Technology and Economics},
	year = {2017}
}
```
