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