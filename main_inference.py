from disambiguator import Disambiguator
from config.config import ModelConfiguration


def main():
    model_configuration = ModelConfiguration()

    disambiguator = Disambiguator(model_configuration)

    disambiguator.create_analysis_window_batch_generator()


if __name__ == '__main__':
    main()