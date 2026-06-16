# Analysis of Texas Tech Football Offense Stats for the 2025 Season
# Jaden Tumale
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
os.makedirs("outputs", exist_ok=True)

# Load and clean the data
df = pd.read_csv("TTUFB_Offense_Stats.csv")
#print("Missing Values:")
#print(df.isnull().sum())
df["Margin"] = df["Pts"] - df["PtsO"]
df = df.iloc[:-1]
#print(df.head())
corr = df.corr(numeric_only=True)
margin_corr = corr["Margin"].sort_values(ascending=False)

# Remove Margin itself
margin_corr = margin_corr.drop("Margin")
margin_corr.to_csv("margin_correlations.csv")
#print(margin_corr)

#Find best and worst correlations
best_corr = margin_corr.nlargest(5)
worst_corr = margin_corr.nsmallest(5)

# Positive correlationss
plt.figure(figsize=(10,6))

best_corr.sort_values().plot(kind="barh")

plt.title("Positive Correlations with Margin of Victory")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("outputs/top_positive_correlations.png")
#plt.show()

# Worst correlations
plt.figure(figsize=(10,6))

worst_corr.sort_values().plot(kind="barh")

plt.title("Negative Correlations with Margin of Victory")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.savefig("outputs/top_negative_correlations.png")
#plt.show()



# Regression Model

X = df[
    ["Avg Yds","Pass Y/A","Rush Y/A","TO","Pnt"]
]

y = df["Margin"]

model = LinearRegression()

model.fit(X, y)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

predictions = model.predict(X)

Xr2 = r2_score(y, predictions)

print(f"R²: {Xr2:.3f}")

plt.figure(figsize=(8,6))

plt.scatter(y, predictions)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--"
)

plt.xlabel("Actual Margin")
plt.ylabel("Predicted Margin")
plt.title("Actual vs Predicted Margin of Victory")

plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted_margin.png")
#plt.show()


# Competing Models
reg1 = df[["Avg Yds"]]
reg2 = df[["Avg Yds", "TO"]]
reg3 = df[["Avg Yds", "TO", "Pnt"]]
m1 = LinearRegression()
model.fit(reg1, y)
m2 = LinearRegression()
model.fit(reg2, y)
m3 = LinearRegression()
model.fit(reg3, y)
m1_coefficients = pd.DataFrame({
    "Feature": reg1.columns,
    "Coefficient": m1.coef_
})
m2_coefficients = pd.DataFrame({
    "Feature": reg2.columns,
    "Coefficient": m2.coef_
})
m3_coefficients = pd.DataFrame({
    "Feature": reg3.columns,
    "Coefficient": m3.coef_
})
print(m1_coefficients)
print(m2_coefficients)
print(m3_coefficients)