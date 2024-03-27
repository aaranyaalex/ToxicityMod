import pandas as pd

df = pd.read_csv("demohistory.csv")

df.set_index("Overall Rating")
num = len(df)
usr = ["user_" + str(num +1)]
df.loc[num] = ["Good", usr[0], 0, 0, 0, 0, 0, 0, 0]
print(df)