# [![Build Status](https://travis-ci.org/peternagy1332/disamorph.svg?branch=master)](https://travis-ci.org/peternagy1332/disamorph) Disamorph
## A Hungarian morphological disambiguator using sequence-to-sequence neural networks

### Usage
```bash
python main_inference.py -m path/to/model/directory
```

### Training in FloydHub
``
floyd run --gpu --env tensorflow-1.3 --data peter.nagy1332/datasets/data/2:data 'ln -s /data /code/data && python main_train.py -dcfg default_configs/floydhub_morpheme_momentum_lstm.yaml -m /output'
``

### Used shell commands during data preprocessing in Szeged Korpusz
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