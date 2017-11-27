import os
import statistics
import operator

maximum_sentence_tokens = []

with open(os.path.join('tokens_in_corpus_by_sentence.csv'), newline='', encoding='utf8') as f:
    token_counts_in_sentence = []
    try:
        tokens = []
        for row in f:
            row = row.rstrip()
            if row == '':
                if len(tokens) > len(maximum_sentence_tokens):
                    maximum_sentence_tokens = tokens
                token_counts_in_sentence.append(len(tokens))
                tokens = []
            else:
                tokens.append(row)

    except KeyboardInterrupt:
        pass

print(" ".join(maximum_sentence_tokens))

print('min:', min(token_counts_in_sentence))
print('max:',max(token_counts_in_sentence))
print('avg:',statistics.mean(token_counts_in_sentence))
print('med:',statistics.median(token_counts_in_sentence))
print('most freq', statistics.mode(token_counts_in_sentence))