from disambiguator import Disambiguator
from config.config import ModelConfiguration


def main():
    model_configuration = ModelConfiguration()

    disambiguator = Disambiguator(model_configuration)

    text = "Már mikor a motoscafon az állomásról befelé hajóztak , és elhagyták a Canale Grandét a rövidebb út kedvéért , Mihálynak feltűntek jobbra és balra a sikátorok ."

    for correct_analyses in  disambiguator.disambiguate_corpus_by_sentence_generator(text):
        print(correct_analyses)

if __name__ == '__main__':
    main()