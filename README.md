# Hungarian morphological disambiguation using sequence-to-sequence neural networks

## Used shell commands during data preprocessing in Szeged corpus
Could be useful for train replicating purposes.

### Acquiring all morphemes from corpus
```bash
cat ./*.new | cut -f5 | grep -oh '^[a-z]*' | sort | uniq > morphemes.txt
```
After all it turned out this command is not necessary since the corpus itself
does not contain any morphemes at all but the roots.

### Acquiring all roots.
```bash
cat ./*.new | cut -f5 | grep -ohEe '^[^[]*' | sort | uniq > roots.txt
```
Hence I collected only the roots.

### Acquiring all possible morphological tags.
```bash
grep -oh '\[[^]]*\]' ./* | sort | uniq -c > tags.txt
```

### Collecting all analyses of words appearing in Szeged corpus
```bash
cat ./* | cut -f1 | sort | uniq | hfst-lookup --cascade=composition ../../emMorph/hfst/hu.hfstol -s | grep . | cut -f1,2 > ../analyses.txt
```

### Non-unique words without correct analysis
```
cat * | cut -f9 | sort | grep '^$' | wc -l
```

### Number of analyses of unique words
```
cat * | grep -e '^.' | cut -f1 | sort | uniq | hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ../../../../programs/emMorph/hfst/hu.hfstol | grep -e '.' | sort | uniq | wc -l
```

### Number of analyses of non-unique words
```
cat * | grep -e '^.' | cut -f1 | hfst-lookup --pipe-mode=input --cascade=composition --xfst=print-pairs --xfst=print-space -s ../../../../programs/emMorph/hfst/hu.hfstol | grep -e '.' | wc -l
```