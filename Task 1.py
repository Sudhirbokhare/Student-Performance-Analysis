import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("StudentsPerformance.csv")


# Basic Information
print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Create Average Score Column

df["average_score"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

# 1. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.show()

# 2. Average Scores by Gender
plt.figure(figsize=(6,4))
sns.barplot(x="gender", y="average_score", data=df)
plt.title("Average Score by Gender")
plt.show()

# 3. Distribution of Math Scores
plt.figure(figsize=(8,5))
sns.histplot(df["math score"], bins=20, kde=True)
plt.title("Math Score Distribution")
plt.show()

# 4. Reading vs Writing Scores
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="reading score",
    y="writing score",
    hue="gender",
    data=df
)
plt.title("Reading Score vs Writing Score")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(
    df[["math score",
        "reading score",
        "writing score",
        "average_score"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# 6. Test Preparation Impact
plt.figure(figsize=(6,4))
sns.barplot(
    x="test preparation course",
    y="average_score",
    data=df
)
plt.title("Impact of Test Preparation Course")
plt.show()

# 7. Lunch Type Impact
plt.figure(figsize=(6,4))
sns.barplot(
    x="lunch",
    y="average_score",
    data=df
)
plt.title("Impact of Lunch Type on Scores")
plt.show()

# 8. Parents Education Impact
plt.figure(figsize=(10,5))
sns.barplot(
    x="parental level of education",
    y="average_score",
    data=df
)
plt.xticks(rotation=45)
plt.title("Parents Education vs Student Performance")
plt.show()

# Top 10 Students
top_students = df.nlargest(10, "average_score")

print("\nTop 10 Students:")
print(top_students[
    ["gender",
     "math score",
     "reading score",
     "writing score",
     "average_score"]
])

# Insights
print("\nINSIGHTS:")
print("1. Students completing test preparation course perform better.")
print("2. Reading and writing scores are highly correlated.")
print("3. Students with standard lunch generally score higher.")
print("4. Higher parental education often relates to better scores.")
print("5. Female students tend to perform better in reading and writing.")
