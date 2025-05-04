# Install Scikit-Learn : pip install scikit-learn

# %%
# Importation des librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Importation des données
df = pd.read_csv('data/creditcard.csv')
print(df.shape)
df.head()

# %%
# Dataset des transactions normales et Dataset des transactions frauduleuses
normal = df[df.Class == 0]
anomaly = df[df.Class == 1]
print("Anomalies :", anomaly.shape)
print("Normal :", normal.shape)

# %%
# Séparation des données en train, validation et test
seed = 2025
X = df.drop(columns=['Class'])
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4, 
    random_state=seed, 
    stratify=y
)

X_test, X_validate, y_test, y_validate = train_test_split(
    X_test, y_test,
    test_size=0.5, 
    random_state=seed, 
    stratify=y_test
)

print("X_train shape, y_train shape :", X_train.shape, y_train.shape)
print("X_validate shape, y_validate shape :", X_validate.shape, y_validate.shape)
print("X_test shape, y_test shape :", X_test.shape, y_test.shape)

print("Class distribution in train set :", y_train.value_counts(normalize=True))
print("Class distribution in validate set :", y_validate.value_counts(normalize=True))
print("Class distribution in test set :", y_test.value_counts(normalize=True))

# %%
# Entraînement du modèle
rf_model = RandomForestClassifier(
    random_state=seed,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# %%
# Accuracy du modèle
eval_acc = rf_model.score(X_validate, y_validate)
print("Validation accuracy :", eval_acc)

# %%
# AUC score
preds = rf_model.predict(X_validate)
auc_score = roc_auc_score(y_validate, preds)
print("Validation AUC score :", auc_score)


# %%
# Courbe ROC
RocCurveDisplay.from_estimator(
    rf_model,
    X_validate,
    y_validate
)

# %%
probs = rf_model.predict_proba(X_validate)[:, 1]
auc_score_with_proba = roc_auc_score(y_validate, probs)
print(auc_score_with_proba)

# %%
# Matrice de confusion
conf_matrix = confusion_matrix(y_validate, preds)
ax = sns.heatmap(conf_matrix, annot=True,fmt='g')
ax.invert_xaxis()
ax.invert_yaxis()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# %%
# K-Fold Cross-Validation

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=seed)
auc_scores = []
model = RandomForestClassifier(
    random_state=seed,
    class_weight='balanced',
    n_jobs=-1
)
auc_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=kf,
    scoring='roc_auc'
)

print("AUC scores for each fold :", auc_scores)
average_auc = np.mean(auc_scores)
print("Average AUC score with K-Fold Cross-Validation :", average_auc)
print("Standard deviation of AUC scores with K-Fold Cross-Validation :", np.std(auc_scores))
print("Best AUC score with K-Fold Cross-Validation :", np.max(auc_scores))





# %%
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=seed)

auc_scores = []
accuracy_scores = []

fold = 1
for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    print(f"\nFold {fold}:")

    model = RandomForestClassifier(
        random_state=seed,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:\n")
    ax = sns.heatmap(cm, annot=True,fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


    print("Accuracy:", acc)
    print("AUC:", auc)
    #print("Confusion Matrix:\n", cm)

    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.title(f"ROC Curve - Fold {fold}")
    plt.show()

    auc_scores.append(auc)
    accuracy_scores.append(acc)
    fold += 1

print("\n==== Résumé sur les 5 folds ====")
print("AUC scores:", auc_scores)
print("Average AUC:", np.mean(auc_scores))
print("Accuracy scores:", accuracy_scores)
print("Average Accuracy:", np.mean(accuracy_scores))

# %%
########### Réglage des hyperparamètres ##########
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
import numpy as np

# Définir le modèle de base
rf = RandomForestClassifier(
    random_state=seed,
    class_weight='balanced',
    n_jobs=-1
)

# Définir l’espace de recherche
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Utiliser StratifiedKFold pour respecter la distribution des classes
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Recherche aléatoire avec priorité à la détection de fraudes (Recall)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=30,  # Nombre d’itérations aléatoires
    scoring=make_scorer(recall_score, pos_label=1),  # Recall sur classe 1
    cv=cv,
    verbose=2,
    random_state=seed,
    n_jobs=-1
)

# Lancer la recherche
random_search.fit(X_train, y_train)

# Résultats
print("Meilleurs hyperparamètres :", random_search.best_params_)
print("Meilleur Recall (fraude détectée) :", random_search.best_score_)
print("Meilleur modèle :", random_search.best_estimator_)

# %%
# https://www.datacamp.com/tutorial/k-fold-cross-validation