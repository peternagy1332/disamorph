from disambiguator import Disambiguator
from config.config import ModelConfiguration
from utils import Utils


def main():
    utils = Utils()
    utils.start_stopwatch()
    utils.redirect_stdout('main-inference')

    model_configuration = ModelConfiguration()
    model_configuration.max_source_sequence_length = 8
    model_configuration.max_target_sequence_length = 14

    disambiguator = Disambiguator(model_configuration)

    text = "Már mikor a motoscafon az állomásról befelé hajóztak , és elhagyták a Canale Grandét a rövidebb út kedvéért , Mihálynak feltűntek jobbra és balra a sikátorok ."

    for correct_analyses in  disambiguator.disambiguate_corpus_by_sentence_generator(text):
        print(correct_analyses)

    utils.stop_stopwatch_and_print_running_time()

if __name__ == '__main__':
    main()