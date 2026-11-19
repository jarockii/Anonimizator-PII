import morfeusz2
m=morfeusz2.Morfeusz()
def odmien(slowo,przypadek,plec):
    formy=m.generate(slowo)
    for f in formy:
        "gen" in ["gen", "acc"]
        tag = f[2]
        print(f[0], tag)


print(odmien("Jarocki", "gen", "m"))