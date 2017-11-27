import os
import statistics
import operator

with open(os.path.join('analyses_in_sentences.csv'), newline='', encoding='utf8') as f:
    analyses_counts_of_sentences = []
    analyses_count_to_sentence_count = {}
    try:
        sentence_analyses_counter = 0
        for row in f:
            row = row.rstrip()
            if row == '':
                analyses_counts_of_sentences.append(sentence_analyses_counter)

                if sentence_analyses_counter in analyses_count_to_sentence_count.keys():
                    analyses_count_to_sentence_count[sentence_analyses_counter]+=1
                else:
                    analyses_count_to_sentence_count.setdefault(sentence_analyses_counter, 1)

                sentence_analyses_counter = 0
            else:
                sentence_analyses_counter+=1

    except KeyboardInterrupt:
        pass

    with open('analyses_count_to_sentence_count.csv', 'w') as f:
        f.writelines(map(lambda a: str(a[0])+'\t'+str(a[1])+os.linesep, sorted(zip(analyses_count_to_sentence_count.keys(), analyses_count_to_sentence_count.values()),key=operator.itemgetter(0))))

print('min:', min(analyses_counts_of_sentences))
print('max:',max(analyses_counts_of_sentences))
print('avg:',statistics.mean(analyses_counts_of_sentences))
print('med:',statistics.median(analyses_counts_of_sentences))
print('most freq:', statistics.mode(analyses_counts_of_sentences))