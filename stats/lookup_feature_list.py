import os
features = []
with open(os.path.join('../', 'data', 'vocabulary_character.tsv'), 'r', encoding='utf-8') as f:
    features.extend(f.read().splitlines())

char_vocabulary = dict(zip(features, range(len(features))))

features = []
with open(os.path.join('../', 'data', 'vocabulary_morpheme.tsv'), 'r', encoding='utf-8') as f:
    features.extend(f.read().splitlines())

morpheme_vocabulary = dict(zip(features, range(len(features))))


def lookup_features_to_ids(vocabulary, feature_list):
    return [vocabulary.get(feature, 4) for feature in feature_list]


print(lookup_features_to_ids(char_vocabulary, ['p', 'á', 'l', 'y', 'a', 'b', 'é', 'r', '[/N]', 'e', 'm', '[Poss.1Sg]', 'e', 't']))
print(lookup_features_to_ids(morpheme_vocabulary, ['pályabér', '[/N]', 'em', '[Poss.1Sg]', 'et']))