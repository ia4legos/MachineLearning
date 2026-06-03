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
