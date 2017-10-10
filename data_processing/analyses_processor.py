import pandas as pd


class AnalysesProcessor(object):
    def build_analyses_dataframe_from_file(self, analyses_path):
        with open(analyses_path, 'r', encoding='utf-8') as f:
            self.__analyses_dataframe = pd.read_csv(f,
                                                    sep='\t',
                                                    header=None,
                                                    names=['word', 'analysis']).groupby(['word'])

    def get_incorrect_analyses_list_for_word(self, word, correct_analysis):
        # TODO: raise exception if dataframe is not built yet
        all_analyses_for_word = self.__analyses_dataframe.get_group(word)
        return list(all_analyses_for_word[all_analyses_for_word['analysis'] != correct_analysis]['analysis'])
