import pandas as pd
import re
import csv


class AnalysesProcessor(object):
    def __init__(self, model_configuration, vocabulary):
        self.__config = model_configuration
        self.__vocabulary = vocabulary

    def build_analyses_dataframe_from_file(self, analyses_path):
        with open(analyses_path, 'r', encoding='utf-8') as f:
            self.__analyses_dataframe = pd.read_csv(f,
                sep='\t', header=None, names=['word', 'analysis'], quoting=csv.QUOTE_NONE)

            self.__analyses_dataframe['analysis'] = self.__analyses_dataframe['analysis']\
                .apply(self.lookup_analysis_to_list)

            self.__analyses_dataframe = self.__analyses_dataframe.groupby(['word'])

    def get_analyses_list_for_word(self, word):
        if word in self.__analyses_dataframe.groups:
            return list(self.__analyses_dataframe.get_group(word)['analysis'])

        if word == '<SOS>': return [[self.__config.marker_start_of_sentence]]

        return [[self.__config.marker_unknown]]

    def get_root_from_analysis(self, analysis):
        root = re.search(r'\w+', analysis, re.UNICODE)
        if root is not None:
            return root.group(0)
        else:
            return None

    def lookup_analysis_to_list(self, analysis):
        if pd.isnull(analysis): return analysis

        root = self.get_root_from_analysis(analysis)
        tags = re.findall(r'\[[^]]*\]', analysis, re.UNICODE)
        return [self.__vocabulary.get(root, self.__config.marker_unknown)] + [
            self.__vocabulary.get(tag, self.__config.marker_unknown) for tag in tags]