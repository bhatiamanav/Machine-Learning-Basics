import pandas as pd

nba1 = pd.read_csv("nba.csv").dropna(how="all")

print(nba1)

#nba1.sort_values(["Salary","Team"],ascending=[True,False],inplace=True)
#nba1["Salary"].fillna(0,inplace=True)
#nba1["College"].fillna("No college",inplace=True)

nba1["Salary"]= nba1["Salary"].fillna(0).astype(int)

#nba1["SalaryRank"] = nba1["Salary"].rank(ascending=False).astype(int)
#nba1.sort_values(by = "Salary",ascending = False,inplace=True)

mask = nba1["Salary"].between(50000,150000)
mask1 = ~mask

print(nba1[mask1])
#print(nba1)

#print(nba1.dropna(subset=["Salary"]))