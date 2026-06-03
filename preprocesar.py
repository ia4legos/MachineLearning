# Commented out IPython magic to ensure Python compatibility.
import numpy as np          # importamos numpy como np
import pandas as pd         # importamos pandas como pd
import math
import random

# Preprocesado
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# División muestras
from sklearn.model_selection import train_test_split


# Función preprocesado
def preprocesar_datos(df, target=None):
    """
    Preprocesa un DataFrame:
      - variables numéricas: imputación por la mediana + estandarización;
      - variables categóricas/booleanas: imputación por la moda + codificación One-Hot.

    Args:
        df (pd.DataFrame): DataFrame a preprocesar.
        target (str, opcional): nombre de la columna objetivo. Si es None (valor por
            defecto), se preprocesan todas las columnas (caso no supervisado).

    Returns:
        pd.DataFrame: DataFrame preprocesado. Si se indicó `target`, se devuelve con
            la columna objetivo añadida al final, sin transformar.

    Nota: el preprocesado se ajusta (fit) con los datos que se le pasan. En aprendizaje
    supervisado, conviene pasar solo el conjunto de entrenamiento (o integrar el
    ColumnTransformer en un Pipeline junto al modelo) para evitar la fuga de información.
    """
    # 1) Separamos el target si lo hay
    dfs = df.copy() if target is None else df.drop(columns=target)

    # 2) Convertimos las booleanas a texto para que el imputador no las trate como
    #    numéricas (mezclar bool con cadenas en un mismo imputador provoca un error)
    bool_features = dfs.select_dtypes(include='bool').columns
    if len(bool_features):
        dfs[bool_features] = dfs[bool_features].astype(str)

    # 3) Identificamos los tipos de variable
    numeric_features = dfs.select_dtypes(include=np.number).columns.tolist()
    categorical_features = dfs.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No hay variables numéricas ni categóricas que preprocesar.")

    # 4) Definimos un transformador por tipo y añadimos solo los que apliquen
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # drop='first' evita la colinealidad (útil en modelos lineales); puedes
            # omitirlo si usas modelos de árboles y prefieres conservar todas las categorías
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first',
                                     sparse_output=False)),
        ]), categorical_features))

    # 5) Combinamos y pedimos la salida directamente como DataFrame de pandas
    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.set_output(transform='pandas')
    df_preprocessed = preprocessor.fit_transform(dfs)

    # Quitamos los prefijos 'num__' / 'cat__' que añade ColumnTransformer a los nombres
    df_preprocessed.columns = [c.split('__', 1)[-1] for c in df_preprocessed.columns]

    # 6) Reincorporamos el target sin transformar (si lo había)
    if target is not None:
        df_preprocessed = pd.concat([df_preprocessed, df[target]], axis=1)

    return df_preprocessed

# Función división de muestras
def split_sample(df, target, size=0.2, semilla=42, estratificar=True):
    """
    Divide un DataFrame en muestras de entrenamiento y test, opcionalmente
    estratificando por la variable objetivo (recomendable en clasificación).

    Parámetros:
    - df (pd.DataFrame): conjunto de datos completo.
    - target (str): nombre de la variable objetivo.
    - size (float): proporción reservada para test, entre 0 y 1 (por defecto 0.2).
    - semilla (int): semilla aleatoria para la reproducibilidad (por defecto 42).
    - estratificar (bool): si True, mantiene las proporciones de clase del target
        en ambas particiones (por defecto True).

    Devuelve:
    - strain, stest (pd.DataFrame): muestras de entrenamiento y test, cada una con
        las predictoras y la columna objetivo.
    """
    # Validaciones básicas
    if target not in df.columns:
        raise KeyError(f"La variable objetivo '{target}' no está en el DataFrame.")
    if not 0 < size < 1:
        raise ValueError(f"`size` debe estar entre 0 y 1 (recibido: {size}).")

    X = df.drop(columns=target)
    y = df[target]

    # Intentamos estratificar; si no es viable (target continuo o clases con un
    # único caso), avisamos y dividimos sin estratificar
    estrato = y if estratificar else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=semilla, stratify=estrato)
        if estratificar:
            print(f"Estratificando por la variable objetivo '{target}'.")
        else:
            print(f"División sin estratificar por '{target}'.")
    except ValueError:
        print(f"Aviso: no es posible estratificar por '{target}' "
              f"(hay clases con un solo caso o el target es continuo). "
              f"Se divide sin estratificar.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, random_state=semilla)

    # Muestras de entrenamiento y test (predictoras + target)
    strain = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    stest = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    print(f"  Entrenamiento: {len(strain)} muestras | Test: {len(stest)} muestras")
    return strain, stest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

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
