from disambiguator import Disambiguator
from config.config import ModelConfiguration


def main():
    model_configuration = ModelConfiguration()

    disambiguator = Disambiguator(model_configuration)

    text="Egy legenda szerint a sajtot egy arab nomád fedezte fel. " \
         "A legenda úgy szól, hogy tejjel töltött meg egy nyeregtáskát, hogy azt fogyassza az úton, " \
         "ameddig keresztüllovagol a sivatagon. Több óra után lovaglás után megállt, hogy szomját oltsa, " \
         "s látta, hogy a tej sápadt vizes folyadékká vált, melyben szilárd fehér darabokban vált ki a sajt. " \
         "A nyeregtáska egy fiatal állat gyomrából készült, ez egy megalvasztó enzimet tartalmazott, amit renninként " \
         "(alvasztóenzim) ismerünk. A tejet valójában a rennin, a forró nap és a ló vágtató mozgásának keveréke " \
         "aludttejjé és savóvá választotta el."

    text = "Már mikor a motoscafon az állomásról befelé hajóztak, és elhagyták a Canale Grandét a rövidebb út kedvéért, Mihálynak feltűntek jobbra és balra a sikátorok."

    disambiguator.disambiguate_text(text)


if __name__ == '__main__':
    main()