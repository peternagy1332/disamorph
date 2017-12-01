import itertools

window_analyses = [
    ['próbál[/V]d[Sbjv.Def.2Sg]'],
    ['meg[/Cnj]', 'meg[/Prev]'],
    ['még[/Adv]'],
    ['egy[/Num]szer[_Mlt-Iter/Adv]', 'egyszer[/Adv]'],
    ['<PUNCT>']
]

window_combinations = list(itertools.product(*window_analyses))


for combination in window_combinations:
    for analysis in combination:
        print(analysis + '<DIV>', end='')
    print()