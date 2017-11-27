import statistics

token_id_to_analyses_count = {}

with open('analyses_of_unique_tokens_by_length.csv', newline='', encoding='utf8') as f:
    analyses_counter_for_token = 0
    for analysis in f:
        analysis = analysis.rstrip()
        if analysis == '':

            token_id_to_analyses_count.setdefault(len(token_id_to_analyses_count), analyses_counter_for_token)

            analyses_counter_for_token=0

        analyses_counter_for_token+=1

print('avg analyses by unique token', statistics.mean(token_id_to_analyses_count.values()))
print('median of analyses by unique token', statistics.median(token_id_to_analyses_count.values()))
print('max analyses by unique token',max(token_id_to_analyses_count.values()))
print('min analyses by unique token', min(token_id_to_analyses_count.values()))
