import os

token_id_to_token_length = {}

with open('unique_tokens_by_length.csv', newline='', encoding='utf8') as f:
    for token in f:
        token_id_to_token_length.setdefault(len(token_id_to_token_length), len(token.rstrip()))

token_length_to_analysis_count = {}

with open('analyses_of_unique_tokens_by_length.csv', newline='', encoding='utf8') as f:
    token_id = 0
    analyses_counter_for_token = 0
    for analysis in f:
        analysis = analysis.rstrip()
        if analysis == '':
            token_length = token_id_to_token_length[token_id]

            if token_length in token_length_to_analysis_count.keys():
                token_length_to_analysis_count[token_length] += analyses_counter_for_token
            else:
                token_length_to_analysis_count.setdefault(token_length, analyses_counter_for_token)

            token_id+=1
            analyses_counter_for_token=0

        analyses_counter_for_token+=1

with open('token_length_to_analyses_count.csv', 'w') as f:
    f.writelines(map(lambda x:str(x[0])+'\t'+str(x[1])+os.linesep,zip(token_length_to_analysis_count.keys(), token_length_to_analysis_count.values())))
