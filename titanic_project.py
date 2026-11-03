import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# =========================================================
# 1. LOAD DATASET + BASIC STATISTICS
# =========================================================

df = pd.read_csv("Titanic-Dataset.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
df.info()

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(df.describe())


# =========================================================
# 2. HANDLE MISSING DATA
# =========================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)
print(df.isnull().sum())

# Age -> Median imputation
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked -> Mode imputation
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has many missing values, so drop it
df = df.drop(columns=["Cabin"])

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)
print(df.isnull().sum())


# =========================================================
# 3. ENCODE CATEGORICAL VARIABLES
# =========================================================

# ---------- Label Encoding: Sex ----------

label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])

# male = 1, female = 0 (depending on encoder's learned order)

print("\n" + "=" * 60)
print("AFTER LABEL ENCODING - SEX")
print("=" * 60)
print(df[["Sex"]].head())


# ---------- One-Hot Encoding: Embarked ----------

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_data = one_hot_encoder.fit_transform(
    df[["Embarked"]]
)

encoded_columns = one_hot_encoder.get_feature_names_out(
    ["Embarked"]
)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns,
    index=df.index
)

# Add encoded columns
df = pd.concat([df, encoded_df], axis=1)

# Remove original Embarked column
df = df.drop(columns=["Embarked"])


# =========================================================
# 4. VISUALIZE AGE DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=30,
    kde=True
)

plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# =========================================================
# 5. SAVE CLEANED DATASET
# =========================================================

df.to_csv(
    "Titanic_Cleaned.csv",
    index=False
)

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Cleaned dataset saved as: Titanic_Cleaned.csv")

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:")
print(df.shape)