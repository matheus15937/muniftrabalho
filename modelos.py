# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# =========================
# 2. LOAD E LIMPEZA
# =========================
df = pd.read_csv('./base_gripe.csv')

df = df.rename(columns={
    'Carimbo de data/hora': 'timestamp',
    'Você ficou gripado no ano passado ?': 'got_flu',
    'Você tomou vacina da gripe no ano passado?': 'vaccinated',
    '  Você frequentou no ano passado,  semanalmente ambientes com muitas pessoas? (salas cheias, ônibus, eventos, etc.)  ': 'crowded_places',
    '  Você viajou no ano passado mais de 100 km de distância?  ': 'traveled',
    '  Você tem alergia nas vias aéreas (rinite, sinusite, etc.)?  ': 'allergy',
    'Quantas horas você dormiu em média por noite no ano passado?': 'sleep_hours',
    'Você praticou atividade física no ano passado?': 'exercise',
    'Você se alimentou de forma balanceada no ano passado?': 'diet',
    'Em média, quantas vezes você lavou as mãos por dia no ano passado?': 'hand_wash',
    'Na sua percepção, o seu nível de estresse no ano passado foi:': 'stress'
})

print("Colunas:")
print(df.columns)


# =========================
# 3. PRÉ-PROCESSAMENTO
# =========================
df = df.drop(columns=['timestamp'])

# tratar missing
df['stress'] = df['stress'].fillna(df['stress'].median())

# target
df['got_flu'] = df['got_flu'].map({'Sim': 1, 'Não': 0})

# categóricas
categorical_cols = df.select_dtypes(include='object').columns.tolist()

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataset após encoding:")
print(df.head())


# =========================
# 4. SPLIT
# =========================
X = df.drop('got_flu', axis=1)
y = df['got_flu']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nDistribuição das classes:")
print(y.value_counts(normalize=True))


# =========================
# 5. ESCALONAMENTO
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 6. FUNÇÃO DE AVALIAÇÃO
# =========================
def avaliar_modelo(nome, y_true, y_pred):
    print(f"\n### {nome} ###")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    print("\nRelatório:")
    print(classification_report(y_true, y_pred))
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_true, y_pred))


# =========================
# 7. MODELOS
# =========================

# ---- KNN ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
avaliar_modelo("KNN", y_test, y_pred_knn)


# ---- NAIVE BAYES ----
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

y_pred_nb = nb.predict(X_test_scaled)
avaliar_modelo("Naive Bayes", y_test, y_pred_nb)


# ---- DECISION TREE ----
dt = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
dt.fit(X_train, y_train)  # árvore NÃO precisa de escala

y_pred_dt = dt.predict(X_test)
avaliar_modelo("Decision Tree", y_test, y_pred_dt)


# =========================
# 8. IMPORTÂNCIA DAS FEATURES
# =========================
importancias = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nImportância das features:")
print(importancias.head(10))


# =========================
# 9. PRISM
# =========================


def prism(X, y):
    rules = []

    X_temp = X.copy().astype(str)

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    covered = set()

    while len(covered) < len(pos_idx):
        best_rule = None
        best_score = 0

        for col in X_temp.columns:
            for val in X_temp[col].unique():

                idx = X_temp[X_temp[col] == val].index

                p = len(set(idx) & set(pos_idx) - covered)
                n = len(set(idx) & set(neg_idx))

                if p + n == 0:
                    continue

                score = p / (p + n)

                if p > 0 and score > best_score:
                    best_score = score
                    best_rule = (col, val)
                    best_idx = set(idx) & set(pos_idx)

        if best_rule is None:
            break

        rules.append((best_rule[0], best_rule[1], best_score))
        covered.update(best_idx)

    return rules


# rodar PRISM
rules = prism(X_train, y_train)


def prism_predict_dual(X, rules_1, rules_0):
    predictions = []

    for _, row in X.iterrows():
        vote_1 = 0
        vote_0 = 0

        for f, v, s in rules_1:
            if str(row[f]) == v:
                vote_1 = max(vote_1, s)

        for f, v, s in rules_0:
            if str(row[f]) == v:
                vote_0 = max(vote_0, s)

        if vote_1 == 0 and vote_0 == 0:
            predictions.append(0)  # default
        elif vote_1 >= vote_0:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions)

rules_1 = prism(X_train, y_train)
rules_0 = prism(X_train, 1 - y_train)

y_pred_prism = prism_predict_dual(X_test, rules_1, rules_0)

avaliar_modelo("PRISM", y_test, y_pred_prism)

# =========================
# PRINT DAS REGRAS
# =========================

print("\n### REGRAS PRISM - CLASSE 1 (GRIPADO) ###")
for f, v, s in rules_1:
    print(f"IF {f} == {v} THEN got_flu = 1 (score={s:.2f})")


print("\n### REGRAS PRISM - CLASSE 0 (NÃO GRIPADO) ###")
for f, v, s in rules_0:
    print(f"IF {f} == {v} THEN got_flu = 0 (score={s:.2f})")