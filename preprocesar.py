# Commented out IPython magic to ensure Python compatibility.
import numpy as np          # importamos numpy como np
import pandas as pd         # importamos pandas como pd
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocesar_datos(df, target=None, preprocessor=None):
    """
    Preprocesa un DataFrame SIN fuga de información.
    - preprocessor=None  -> AJUSTA con `df` (train), transforma y devuelve (df_prep, preprocessor).
    - preprocessor dado   -> solo TRANSFORMA `df` (test) con lo aprendido del train.
    Devuelve SIEMPRE una tupla (df_preprocesado, preprocessor).
    """
    dfs = df.copy() if target is None else df.drop(columns=target)

    # Las booleanas se tratan como texto (consistente en train y test)
    bool_features = dfs.select_dtypes(include='bool').columns
    if len(bool_features):
        dfs[bool_features] = dfs[bool_features].astype(str)

    if preprocessor is None:
        # AJUSTE (con la muestra de entrenamiento)
        numeric_features = dfs.select_dtypes(include=np.number).columns.tolist()
        categorical_features = dfs.select_dtypes(include=['object', 'category']).columns.tolist()
        if not numeric_features and not categorical_features:
            raise ValueError("No hay variables numéricas ni categóricas que preprocesar.")
        transformers = []
        if numeric_features:
            transformers.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numeric_features))
        if categorical_features:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first',
                                         sparse_output=False))]), categorical_features))
        preprocessor = ColumnTransformer(transformers=transformers)
        preprocessor.set_output(transform='pandas')
        df_prep = preprocessor.fit_transform(dfs)
    else:
        # TRANSFORMACIÓN (con test, usando lo aprendido del train)
        df_prep = preprocessor.transform(dfs)

    # Quitamos los prefijos 'num__' / 'cat__'
    df_prep.columns = [c.split('__', 1)[-1] for c in df_prep.columns]

    # Reincorporamos el target sin transformar
    if target is not None:
        df_prep = pd.concat([df_prep, df[target]], axis=1)

    return df_prep, preprocessor

def split_sample(df, target, size=0.2, semilla=42, estratificar=True):
    """
    Divide un DataFrame en entrenamiento y test, opcionalmente estratificando por el target.
    Devuelve (strain, stest).
    """
    if target not in df.columns:
        raise KeyError(f"La variable objetivo '{target}' no está en el DataFrame.")
    if not 0 < size < 1:
        raise ValueError(f"`size` debe estar entre 0 y 1 (recibido: {size}).")

    X = df.drop(columns=target)
    y = df[target]
    estrato = y if estratificar else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=semilla, stratify=estrato)
        print(f"{'Estratificando' if estratificar else 'División sin estratificar'} por '{target}'.")
    except ValueError:
        print(f"Aviso: no es posible estratificar por '{target}' "
              f"(hay clases con un solo caso o el target es continuo). Se divide sin estratificar.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=semilla)

    strain = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    stest = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    print(f"  Entrenamiento: {len(strain)} muestras | Test: {len(stest)} muestras")
    return strain, stest

def descriptivo_target_predictoras(df, target, ncols=3, ancho=5, alto=4):
    """
    Representa el target frente a cada predictora:
      - numéricas  -> diagrama de caja por clase del target
      - categóricas -> barras con el % de cada clase del target dentro de la categoría
    """
    predictoras = [c for c in df.columns if c != target]
    numericas = df[predictoras].select_dtypes(include=np.number).columns.tolist()

    n = len(predictoras)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ancho * ncols, alto * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, var in zip(axes, predictoras):
        if var in numericas:
            # Numérica: diagrama de caja según la clase del target
            sns.boxplot(data=df, x=target, y=var, hue=target, legend=False, ax=ax)
            ax.set_title(f'{var} según {target}')
        else:
            # Categórica: % de cada clase del target dentro de cada categoría
            ct = pd.crosstab(df[var], df[target], normalize='index').mul(100)
            ct.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel('Porcentaje (%)')
            ax.set_xlabel(var)
            ax.set_title(f'{var} vs {target}')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title=target, fontsize=8)
            ax.margins(y=0.12)
            for contenedor in ax.containers:
                ax.bar_label(contenedor, fmt='%.0f%%', fontsize=7, padding=1)

    # Ocultamos los ejes sobrantes de la cuadrícula
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, BaggingClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def _scores_clase_positiva(clf, X):
    """Devuelve la puntuación de la clase positiva para la AUC.
    Usa predict_proba si existe; si no, decision_function; si ninguna, None."""
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        idx = list(clf.classes_).index(1)      # columna de la clase positiva (1)
        return proba[:, idx]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return None


def comparar_clasificador_2cls(strain, target, sizeval=0.3, semilla=42,
                               models_to_train=None, pos_label=None):
    """
    Entrena varios modelos de clasificación binaria y devuelve sus métricas.

    Args:
        strain (pd.DataFrame): conjunto de entrenamiento (predictoras ya
            preprocesadas a numérico + columna objetivo).
        target (str): nombre de la columna objetivo (debe tener 2 clases).
        sizeval (float): fracción del conjunto reservada para validación.
        semilla (int): semilla de aleatorización (reproducibilidad).
        models_to_train (list, opcional): nombres de modelos a entrenar; si es
            None se entrenan todos.
        pos_label (opcional): clase considerada "positiva". Si es None, se
            elige automáticamente (1 si las clases son 0/1; 'Yes' si existe;
            en otro caso, la clase minoritaria).

    Returns:
        pd.DataFrame: métricas por modelo (Accuracy, Balanced_Accuracy,
            Precision, Recall, Specificity, F1, AUC), indexado por algoritmo.
    """
    # Conjunto completo de clasificadores (versiones por defecto)
    all_classifiers = {
        "lr": LogisticRegression(random_state=semilla, max_iter=1000),
        "ridge": RidgeClassifier(random_state=semilla),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "svc": SVC(kernel='linear', random_state=semilla),
        "rbf": SVC(kernel='rbf', random_state=semilla),
        "dt": DecisionTreeClassifier(random_state=semilla),
        "rf": RandomForestClassifier(random_state=semilla),
        "extra": ExtraTreesClassifier(random_state=semilla),
        "bagging": BaggingClassifier(random_state=semilla),
        "ada": AdaBoostClassifier(random_state=semilla),
        "gbc": GradientBoostingClassifier(random_state=semilla),
        "hgb": HistGradientBoostingClassifier(random_state=semilla),
        "lightgbm": LGBMClassifier(random_state=semilla, verbose=-1),
        "xgboost": XGBClassifier(random_state=semilla, eval_metric='logloss'),
    }

    # Selección de los modelos a entrenar
    if models_to_train is None:
        classifiers = all_classifiers
    else:
        classifiers = {n: all_classifiers[n] for n in models_to_train
                       if n in all_classifiers}
        if len(classifiers) != len(models_to_train):
            print("Advertencia: algunos nombres de modelos no son válidos.")

    # División entrenamiento/validación (sizeval = fracción de validación)
    strain_df, sval_df = split_sample(strain, target, sizeval, semilla, True)
    X_train, y_train = strain_df.drop(columns=target), strain_df[target]
    X_val, y_val = sval_df.drop(columns=target), sval_df[target]

    # Comprobación de que el problema es binario y elección de la clase positiva
    clases = sorted(pd.unique(y_train), key=str)
    if len(clases) != 2:
        raise ValueError(f"Esta función es para 2 clases; encontradas "
                         f"{len(clases)}: {clases}")
    if pos_label is None:
        if set(clases) == {0, 1}:
            pos_label = 1
        elif 'Yes' in clases:
            pos_label = 'Yes'
        else:
            pos_label = y_val.value_counts().idxmin()   # clase minoritaria
    print(f"Clase positiva: {pos_label!r}")

    # Codificamos la respuesta a 0/1 (positiva = 1). Esto unifica el cálculo de
    # métricas y evita el error de XGBoost con etiquetas de texto.
    y_train01 = (y_train == pos_label).astype(int)
    y_val01 = (y_val == pos_label).astype(int)

    # Entrenamiento y cálculo de métricas sobre la validación
    results = []
    for name, clf in classifiers.items():
        print(f"Entrenando {name}...")
        try:
            clf.fit(X_train, y_train01)
            y_pred = clf.predict(X_val)

            # Métricas con las funciones de scikit-learn (positiva = 1).
            # La especificidad (TNR) es el recall de la clase negativa (pos_label=0).
            accuracy = accuracy_score(y_val01, y_pred)
            balanced_acc = balanced_accuracy_score(y_val01, y_pred)
            precision = precision_score(y_val01, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_val01, y_pred, pos_label=1, zero_division=0)        # sensibilidad (TPR)
            specificity = recall_score(y_val01, y_pred, pos_label=0, zero_division=0)   # especificidad (TNR)
            f1 = f1_score(y_val01, y_pred, pos_label=1, zero_division=0)

            # AUC: requiere scores/probabilidades, no las predicciones duras
            scores = _scores_clase_positiva(clf, X_val)
            auc = roc_auc_score(y_val01, scores) if scores is not None else None

            results.append({'Algorithm': name, 'Accuracy': accuracy,
                            'Balanced_Accuracy': balanced_acc,
                            'Precision': precision, 'Recall': recall,
                            'Specificity': specificity, 'F1': f1, 'AUC': auc})
        except Exception as e:
            print(f"  Error entrenando {name}: {e}")
            results.append({'Algorithm': name, 'Accuracy': None,
                            'Balanced_Accuracy': None, 'Precision': None,
                            'Recall': None, 'Specificity': None, 'F1': None,
                            'AUC': None})

    return pd.DataFrame(results).set_index('Algorithm')


"""
Comparación de modelos de clasificación multiclase.

Requiere tener definida/importada la función `split_sample`.
LightGBM y XGBoost se instalan con `pip install lightgbm xgboost` si no están.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, BaggingClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def _scores_multiclase(clf, X):
    """Scores por clase para la AUC multiclase (deben sumar 1).
    Usa predict_proba; si no existe, aplica softmax a decision_function."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        d = np.atleast_2d(clf.decision_function(X))
        e = np.exp(d - d.max(axis=1, keepdims=True))   # softmax -> las filas suman 1
        return e / e.sum(axis=1, keepdims=True)
    return None


def comparar_clasificador_multicls(strain, target, sizeval=0.3, semilla=42,
                                   models_to_train=None):
    """
    Entrena varios modelos de clasificación multiclase y devuelve sus métricas.

    Args:
        strain (pd.DataFrame): conjunto de entrenamiento (predictoras ya
            preprocesadas a numérico + columna objetivo).
        target (str): nombre de la columna objetivo.
        sizeval (float): fracción del conjunto reservada para validación.
        semilla (int): semilla de aleatorización (reproducibilidad).
        models_to_train (list, opcional): nombres de modelos a entrenar; si es
            None se entrenan todos.

    Returns:
        pd.DataFrame: métricas por modelo (Accuracy, Balanced_Accuracy,
            Precision, Recall, F1, AUC), indexado por algoritmo. Precisión,
            sensibilidad, F1 y AUC se promedian con `average='weighted'`
            (la AUC en modo One-vs-Rest).
    """
    # Conjunto completo de clasificadores (versiones por defecto)
    all_models = {
        "lr": LogisticRegression(random_state=semilla, max_iter=1000),
        "ridge": RidgeClassifier(random_state=semilla),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "svc": SVC(kernel='linear', random_state=semilla),
        "rbf": SVC(kernel='rbf', random_state=semilla),
        "dt": DecisionTreeClassifier(random_state=semilla),
        "rf": RandomForestClassifier(random_state=semilla),
        "extra": ExtraTreesClassifier(random_state=semilla),
        "bagging": BaggingClassifier(random_state=semilla),
        "ada": AdaBoostClassifier(random_state=semilla),
        "gbc": GradientBoostingClassifier(random_state=semilla),
        "hgb": HistGradientBoostingClassifier(random_state=semilla),
        "lightgbm": LGBMClassifier(random_state=semilla, verbose=-1),
        "xgboost": XGBClassifier(random_state=semilla, eval_metric='mlogloss'),
    }

    # Selección de modelos
    if models_to_train is None:
        models = all_models
    else:
        models = {n: all_models[n] for n in models_to_train if n in all_models}
        if len(models) != len(models_to_train):
            print("Advertencia: algunos nombres de modelos no son válidos.")

    # División entrenamiento/validación (sizeval = fracción de validación)
    strain_df, sval_df = split_sample(strain, target, sizeval, semilla, True)
    X_train, y_train = strain_df.drop(columns=target), strain_df[target]
    X_val, y_val = sval_df.drop(columns=target), sval_df[target]

    # Codificamos la respuesta a 0..K-1. Necesario para XGBoost (no admite
    # etiquetas de texto) y para alinear las columnas de probabilidad en la AUC.
    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    etiquetas = np.arange(len(le.classes_))   # por si alguna clase falta en validación

    # Entrenamiento y métricas sobre la validación
    results = []
    for name, model in models.items():
        print(f"Entrenando {name}...")
        try:
            model.fit(X_train, y_train_enc)
            y_pred = model.predict(X_val)

            # Métricas con scikit-learn (promedio ponderado para multiclase)
            accuracy = accuracy_score(y_val_enc, y_pred)
            balanced_acc = balanced_accuracy_score(y_val_enc, y_pred)
            precision = precision_score(y_val_enc, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val_enc, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val_enc, y_pred, average='weighted', zero_division=0)

            # AUC multiclase One-vs-Rest (necesita scores/probabilidades)
            try:
                scores = _scores_multiclase(model, X_val)
                auc = (roc_auc_score(y_val_enc, scores, multi_class='ovr',
                                     average='weighted', labels=etiquetas)
                       if scores is not None else None)
            except Exception:
                auc = None

            results.append({'Algorithm': name, 'Accuracy': accuracy,
                            'Balanced_Accuracy': balanced_acc,
                            'Precision': precision, 'Recall': recall,
                            'F1': f1, 'AUC': auc})
        except Exception as e:
            print(f"  Error entrenando {name}: {e}")
            results.append({'Algorithm': name, 'Accuracy': None,
                            'Balanced_Accuracy': None, 'Precision': None,
                            'Recall': None, 'F1': None, 'AUC': None})

    return pd.DataFrame(results).set_index('Algorithm')
