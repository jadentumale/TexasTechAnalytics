import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

# Load and clean the data
df = pd.read_csv("TTUFB_Offense_Stats.csv")
print("Missing Values:")
print(df.isnull().sum())
df["Margin"] = df["Pts"] - df["PtsO"]
df = df.iloc[:-1]

print(df.head())
print(df.describe())

corr = df.corr(numeric_only=True)

margin_corr = corr["Margin"].sort_values(ascending=False)

# Remove Margin itself
margin_corr = margin_corr.drop("Margin")

best_corr = margin_corr.nlargest(5)
worst_corr = margin_corr.nsmallest(5)

print(worst_corr)

print(margin_corr)
margin_corr.to_csv("margin_correlations.csv")

# Positive correlationss
plt.figure(figsize=(10,6))

best_corr.sort_values().plot(kind="barh")

plt.title("Positive Correlations with Margin of Victory")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("outputs/top_positive_correlations.png")
plt.show()

# Worst correlations
plt.figure(figsize=(10,6))

worst_corr.sort_values().plot(kind="barh")

plt.title("Negative Correlations with Margin of Victory")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("outputs/top_negative_correlations.png")
plt.show()

# Scatterplot for 