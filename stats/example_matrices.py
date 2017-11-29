import numpy as np
import os

np.set_printoptions(linewidth=200, precision=2)

features = []
with open(os.path.join('../', 'data', 'vocabulary_character.tsv'), 'r', encoding='utf-8') as f:
    features.extend(f.read().splitlines())

char_vocabulary = dict(zip(features, range(len(features))))
char_inverse_vocabulary = {v: k for k, v in char_vocabulary.items()}

features = []
with open(os.path.join('../', 'data', 'vocabulary_morpheme.tsv'), 'r', encoding='utf-8') as f:
    features.extend(f.read().splitlines())

morpheme_vocabulary = dict(zip(features, range(len(features))))
morpheme_inverse_vocabulary = {v: k for k, v in morpheme_vocabulary.items()}


def lookup_ids_to_features(inverse_vocabulary, ids_list):
    return [inverse_vocabulary.get(id, "<PAD>") for id in ids_list]

si = np.load(os.path.join('../', 'data', 'train_matrices', 'morpheme', 'source_input', 'sentence_source_input_examples_0000000000.npy'))[:5,:16]
ti = np.load(os.path.join('../', 'data', 'train_matrices', 'morpheme', 'target_input', 'sentence_target_input_examples_0000000000.npy'))[:5,:5]
to = np.load(os.path.join('../', 'data', 'train_matrices', 'morpheme', 'target_output', 'sentence_target_output_examples_0000000000.npy'))[:5,:5]

s_vpad = np.zeros((2,si.shape[1]),dtype=np.int32)
si = np.concatenate((si,s_vpad),axis=0)

t_vpad = np.zeros((2,ti.shape[1]))
ti = np.concatenate((ti,t_vpad),axis=0)
to = np.concatenate((to,t_vpad),axis=0)

print('Source input')
for row in si:
    print(" ".join(map(lambda x: str(int(x)),row.tolist())))
for row in si:
    print(" ".join(lookup_ids_to_features(morpheme_inverse_vocabulary, row)))

print('Target input')
for row in ti:
    print(" ".join(map(lambda x: str(int(x)), row.tolist())))
for row in ti:
    print(" ".join(lookup_ids_to_features(morpheme_inverse_vocabulary, row)))

print('Target output')
for row in to:
    print(" ".join(map(lambda x: str(int(x)), row.tolist())))
for row in to:
    print(" ".join(lookup_ids_to_features(morpheme_inverse_vocabulary, row)))