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


print(lookup_features_to_ids(morpheme_vocabulary, ['próbál','[/V]', 'd', '[Sbjv.Def.2Sg]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['meg', '[/Cnj]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['meg', '[/Prev]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['még', '[/Adv]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['egy', '[/Num]', 'szer', '[_Mlt-Iter/Adv]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['egyszer', '[/Adv]']))
print(lookup_features_to_ids(morpheme_vocabulary, ['.']))