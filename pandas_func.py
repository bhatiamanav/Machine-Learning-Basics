import pandas as pd 

bond = pd.read_csv("jamesbond.csv",index_col="Film")
bond.sort_index(inplace=True)

#bond.loc["Dr. No","Actor"] = "Sir Sean Connery"

mask= bond["Actor"] == "Sean Connery"
bond.loc[mask,"Actor"] = "Sir Sean Connery"

#bond.rename(columns = {"Year":"Release Date","Box Office":"Earnings"},inplace=True)

#bond.drop("Casino Royale",inplace = True)

#actor = bond.pop("Actor")
#print(bond)
#print(actor)

#bond.columns = [column_name.replace(" ","_") for column_name in bond.columns]
#print(bond.query("Actor=='Roger Moore' or Director=='John Glen'"))

def add_millions(number):
    return str(number) + " Millions!"

columns = ["Box Office","Budget","Bond Actor Salary"]

for col in columns:
    bond[col] = bond[col].apply(add_millions)

print(bond.head())
#print(bond["Box Office"].apply(add_millions))