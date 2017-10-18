from disambiguator import Disambiguator
from config.config import ModelConfiguration


def main():
    model_configuration = ModelConfiguration()

    disambiguator = Disambiguator(model_configuration)


    for windows_combination_heights_in_sentence, logits_in_sentence in disambiguator.analysis_window_batches_to_logits_generator():
        print('SENTENCE LOGITS')
        print('windows_combination_heights_in_sentence', windows_combination_heights_in_sentence)

        logits_for_windows = []
        current_window_logits_of_combinations = []
        current_window_height = windows_combination_heights_in_sentence[0]
        windows_combination_heights_in_sentence = windows_combination_heights_in_sentence[1:]

        for logit in logits_in_sentence:
            current_window_logits_of_combinations.append(logit)

            if current_window_height > 1:
                current_window_height-=1
            else:
                if len(windows_combination_heights_in_sentence) > 0:
                    logits_for_windows.append(current_window_logits_of_combinations)
                    print('\t', current_window_logits_of_combinations)
                    current_window_logits_of_combinations = []
                    current_window_height = windows_combination_heights_in_sentence[0]
                    windows_combination_heights_in_sentence = windows_combination_heights_in_sentence[1:]
                else:
                    break

        print('-'*70)


if __name__ == '__main__':
    main()