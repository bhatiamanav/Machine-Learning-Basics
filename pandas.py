import pandas as pd 

#pokemon = pd.read_csv("pokemon.csv")
pokemon = pd.read_csv("pokemon.csv",squeeze=True,usecols=["Name"])
pokemon["Cartoon"] = "Pokemon"

print(pokemon.head())


