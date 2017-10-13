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
                sep='\t', header=None, names=['word', 'analysis'], quoting=csv.QUOTE_NONE).groupby(['word'])

    def get_analyses_list_for_word(self, word):
        # TODO: raise exception if dataframe is not built yet
        if word in self.__analyses_dataframe.groups:
            return list(self.__analyses_dataframe.get_group(word)['analysis'])

        # This should be UNK value later
        return ['MARKER_UNKNOWN']

    def lookup_analysis_to_list(self, analysis):
        if pd.isnull(analysis): return analysis

        root = re.search(r'\w+', analysis, re.UNICODE).group(0)
        tags = re.findall(r'\[[^]]*\]', analysis, re.UNICODE)
        return [self.__vocabulary.get(root, self.__config.marker_unknown)] + [
            self.__vocabulary.get(tag, self.__config.marker_unknown) for tag in tags]