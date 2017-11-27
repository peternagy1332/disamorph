import statistics
import numpy as np

token_to_id = {}

token_id_to_total_analyses_count = {}

token_id_to_analyses_count = {}
token_id_to_frequency = {}

total_word_count = 0

current_token_id = None
with open('token_with_analysis_in_corpus.csv', encoding='utf8') as f:
    analyses_count = 0
    for row in f:
        row = row.rstrip()

        if row == '':
            token_id_to_analyses_count.setdefault(current_token_id, analyses_count)
            current_token_id = None
            analyses_count=0
            continue

        row = row.split('\t')

        token = row[0]
        analysis = row[1]

        token_to_id.setdefault(token, len(token_to_id))

        token_id = token_to_id[token]

        if token_id in token_id_to_total_analyses_count.keys():
            token_id_to_total_analyses_count[token_id]+=1
        else:
            token_id_to_total_analyses_count.setdefault(token_id,1)

        if current_token_id is None:
            current_token_id = token_id
            total_word_count+=1

            if token_id in token_id_to_frequency.keys():
                token_id_to_frequency[token_id]+=1
            else:
                token_id_to_frequency.setdefault(token_id, 1)

            frequency_incremented = True

        analyses_count += 1


for token_id, total_analyses_count in token_id_to_total_analyses_count.items():
    if token_id_to_frequency[token_id]*token_id_to_analyses_count[token_id] != total_analyses_count:
        print(token_id)

print('total_word_count', total_word_count)
print('total_analysis_count', sum(token_id_to_total_analyses_count.values()))

print('avg')
print(sum(token_id_to_total_analyses_count.values())/total_word_count)
print(sum(token_id_to_total_analyses_count.values())/sum(token_id_to_frequency.values()))
print('med', statistics.median(token_id_to_analyses_count.values()))
print('most freq ana count', statistics.mode(token_id_to_analyses_count.values()))
print('most freq ana frq', statistics.mode(token_id_to_frequency.values()))