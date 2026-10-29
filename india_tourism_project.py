import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, r2_score, mean_absolute_error, mean_squared_error, ConfusionMatrixDisplay)

os.makedirs("figures", exist_ok=True)
cv_reg = KFold(5, shuffle=True, random_state=42)
cv_cls = StratifiedKFold(5, shuffle=True, random_state=42)

def show(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.show()
    plt.close()

df = pd.read_csv("C:\\Users\\Aman Rana\\OneDrive\\Desktop\\predicive project\\India_Tourism_Statistics_2024_Table_2.4.4.csv")
print(df.shape)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df.columns = ["Sl", "Country", "Arrivals", "North", "South", "West_Central", "East", "North_East"]
df["Country"] = df["Country"].str.strip()
df = df[df["Country"] != "Grand Total"]
df["Sl"] = df["Sl"].astype(int)

continent_start = {1: "Africa", 57: "Americas", 103: "Asia-Pacific", 154: "Europe", 214: "Middle East"}
group_start = {1: "Africa", 2: "North Africa", 7: "Sub-Saharan Africa", 58: "Caribbean", 78: "Central America",
               86: "North America", 90: "South America", 104: "North-East Asia", 114: "Oceania",
               133: "South Asia", 142: "South-East Asia", 155: "Central/Eastern Europe",
               177: "Northern Europe", 185: "Southern Europe", 204: "Western Europe", 214: "Middle East"}
find = lambda n, start: start[max(k for k in start if k <= n)]
df["Continent"] = df["Sl"].apply(lambda n: find(n, continent_start))
df["Sub_Region"] = df["Sl"].apply(lambda n: find(n, group_start))

group_rows = ["Africa", "North Africa", "Sub-Saharan Africa", "Americas", "Caribbean", "Central America",
              "North America", "South America", "Asia and the Pacific", "North-East Asia", "Oceania",
              "South Asia", "South-East Asia", "Europe", "Central/Eastern Europe", "Northern Europe",
              "Southern/Medit. Europe", "Western Europe", "Middle East", "Others Not Classified Elsewhere"]
df = df[~df["Country"].isin(group_rows)].drop(columns="Sl").reset_index(drop=True)
print("Countries kept:", len(df))

shares = ["North", "South", "West_Central", "East", "North_East"]
print("Row sums of shares:\n", df[shares].sum(axis=1).describe().round(2))
df["Log_Arrivals"] = np.log10(df["Arrivals"])
df["High_Volume"] = (df["Arrivals"] >= df["Arrivals"].median()).astype(int)
print(df["High_Volume"].value_counts())
print(df[["Arrivals", "Log_Arrivals"] + shares].describe().round(2))
print(df[["Arrivals", "Log_Arrivals"] + shares].skew().round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Arrivals"], bins=30, ax=ax[0]).set_title("Arrivals (raw)")
sns.histplot(df["Log_Arrivals"], bins=30, kde=True, ax=ax[1]).set_title("Arrivals (log10)")
show("01_arrivals_distribution")

top10 = df.nlargest(10, "Arrivals").sort_values("Arrivals")
plt.figure(figsize=(8, 5))
plt.barh(top10["Country"], top10["Arrivals"], color="orange")
plt.title("Top 10 Countries by Arrivals")
show("02_top10_countries")

df[shares].mean().plot(kind="bar", figsize=(7, 4), color="steelblue", rot=0, title="Average Share of Visits by Indian Region")
show("03_average_regional_share")

df.groupby("Continent")["Arrivals"].sum().sort_values().plot(kind="barh", figsize=(8, 4), color="purple", title="Total Arrivals by Continent")
show("04_continent_arrivals")

df.groupby("Continent")[shares].mean().plot(kind="bar", stacked=True, figsize=(9, 5), rot=15, title="Regional Preference by Continent")
show("05_continent_preference")

plt.figure(figsize=(8, 6))
sns.heatmap(df[["Log_Arrivals"] + shares].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
show("06_correlation_heatmap")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["North", "South", "East"]):
    sns.boxplot(data=df, x="High_Volume", y=c, ax=a).set_title(f"{c} share by market size")
show("07_boxplots_market_size")

for c in shares:
    print(c, "Pearson r with Log_Arrivals: %.3f (p=%.4f)" % stats.pearsonr(df[c], df["Log_Arrivals"]))
for c in shares:
    u, p = stats.mannwhitneyu(df[df.High_Volume == 1][c], df[df.High_Volume == 0][c])
    print(c, "Mann-Whitney p = %.4f" % p)
for c in shares:
    h, p = stats.kruskal(*[g[c] for _, g in df.groupby("Continent")])
    print(c, "Kruskal-Wallis across continents p = %.4f" % p)
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df["Continent"], df["High_Volume"]))
print("Chi-square continent vs market size: chi2=%.2f, p=%.4f" % (chi2, p))

xs = df[["South"]]
ys = df["North"]
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, random_state=42)
slr = LinearRegression().fit(xs_train, ys_train)
ys_pred = slr.predict(xs_test)
slr_r2 = r2_score(ys_test, ys_pred)
slr_mae = mean_absolute_error(ys_test, ys_pred)
slr_rmse = np.sqrt(mean_squared_error(ys_test, ys_pred))
slr_cv = cross_val_score(LinearRegression(), xs, ys, cv=cv_reg, scoring="r2").mean()
print("SLR: North = %.2f + (%.3f) x South" % (slr.intercept_, slr.coef_[0]))
print("R2=%.3f  MAE=%.2f  RMSE=%.2f  CV R2=%.3f" % (slr_r2, slr_mae, slr_rmse, slr_cv))

plt.figure(figsize=(7, 5))
plt.scatter(xs_train, ys_train, alpha=0.5, label="Train")
plt.scatter(xs_test, ys_test, color="red", label="Test")
line = pd.DataFrame({"South": [0, 100]})
plt.plot(line, slr.predict(line), color="black", label="Regression line")
plt.xlabel("South share (%)")
plt.ylabel("North share (%)")
plt.legend()
plt.title("Simple Linear Regression")
show("08_slr_fit")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(ys_pred, ys_test - ys_pred)
ax[0].axhline(0, color="red")
ax[0].set_title("Residuals vs Predicted")
ax[1].scatter(ys_test, ys_pred)
ax[1].plot([0, 100], [0, 100], color="red")
ax[1].set_title("Actual vs Predicted")
show("09_slr_residuals")

X = pd.concat([pd.get_dummies(df["Sub_Region"]).astype(int), df[shares]], axis=1)
y = df["High_Volume"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

k_values = list(range(1, 16, 2))
k_scores = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), X_train, y_train, cv=cv_cls).mean() for k in k_values]
best_k = k_values[int(np.argmax(k_scores))]
depths = list(range(1, 9))
d_scores = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X_train, y_train, cv=cv_cls).mean() for d in depths]
best_depth = depths[int(np.argmax(d_scores))]
print("Best k:", best_k, "| Best depth:", best_depth)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(k_values, k_scores, marker="o")
ax[0].set_title("KNN: CV accuracy vs k")
ax[0].set_xlabel("k")
ax[1].plot(depths, d_scores, marker="o", color="green")
ax[1].set_title("Decision Tree: CV accuracy vs depth")
ax[1].set_xlabel("max_depth")
show("10_tuning_knn_tree")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(best_k)),
    "Decision Tree": DecisionTreeClassifier(max_depth=best_depth, random_state=42),
    "Naive Bayes": GaussianNB(),
}

rows, proba = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba[name] = model.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "CV Accuracy": cross_val_score(model, X_train, y_train, cv=cv_cls).mean(),
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1 Score": f1_score(y_test, pred),
                 "ROC AUC": roc_auc_score(y_test, proba[name])})
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Low", "High"], cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    show("cm_" + name.lower().replace(" ", "_"))

results = pd.DataFrame(rows).round(3).sort_values("Accuracy", ascending=False)
print(results.to_string(index=False))

plt.figure(figsize=(7, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, proba[name])
    plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, proba[name]):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curves")
show("11_roc_curves")

results.set_index("Model")[["CV Accuracy", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]].plot(kind="bar", figsize=(11, 5), rot=0)
plt.ylim(0, 1)
plt.title("Model Comparison")
show("12_model_comparison")

coef = pd.Series(models["Logistic Regression"][-1].coef_[0], index=X.columns).sort_values()
pd.concat([coef.head(6), coef.tail(6)]).plot(kind="barh", figsize=(8, 5), color="teal", title="Logistic Regression: Strongest Coefficients")
show("13_lr_coefficients")

tree = models["Decision Tree"]
pd.Series(tree.feature_importances_, index=X.columns).nlargest(8).sort_values().plot(kind="barh", figsize=(8, 4), color="green", title="Decision Tree Feature Importance")
show("14_tree_importance")

plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=list(X.columns), class_names=["Low", "High"], filled=True, max_depth=3, fontsize=8)
show("15_tree_structure")

plt.figure(figsize=(7, 4))
plt.hist(proba["Naive Bayes"][y_test == 0], alpha=0.6, label="Actual Low")
plt.hist(proba["Naive Bayes"][y_test == 1], alpha=0.6, label="Actual High")
plt.xlabel("Predicted probability of High volume")
plt.legend()
plt.title("Naive Bayes: Predicted Probabilities")
show("16_nb_probabilities")

print("Best model:", results.iloc[0]["Model"])
