import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

sns.set_theme(style="whitegrid")   # formato común: fondo blanco


def var_contrib(solcp, X):
    """
    Obtiene los loadings de cada variable y las coordenadas (scores) de los sujetos.

    Parámetros:
    - solcp: objeto PCA ya ajustado (solución de componentes principales)
    - X: DataFrame de datos usado en el ajuste de la solución CP

    Devuelve:
    - loadings: DataFrame de pesos (con columna 'Variable')
    - coordenadas: DataFrame de coordenadas de los sujetos en las CP
    """
    nombres = [f'CP{i+1}' for i in range(solcp.n_components_)]
    loadings = pd.DataFrame(solcp.components_.T, columns=nombres, index=X.columns)
    loadings['Variable'] = X.columns
    coordenadas = pd.DataFrame(solcp.transform(X), columns=nombres, index=X.index)
    return loadings, coordenadas


def plot_var_explained(solcp, lx, ly):
    """
    Tabla y gráfico de la proporción de varianza explicada por cada componente.

    Parámetros:
    - solcp: objeto PCA ya ajustado
    - lx, ly: ancho y alto del gráfico
    """
    componentes = np.arange(solcp.n_components_) + 1
    cumVar = pd.DataFrame(np.cumsum(solcp.explained_variance_ratio_) * 100, columns=["Varianza acumulada"])
    expVar = pd.DataFrame(100 * solcp.explained_variance_ratio_, columns=["Varianza explicada"])
    pd.options.display.float_format = '{:.4f}'.format
    tabla = pd.concat([expVar, cumVar], axis=1).set_index(componentes)
    print("Tabla de variabilidad explicada")
    print("==============================")
    print(tabla)

    fig, ax = plt.subplots(figsize=(lx, ly))
    barras = ax.bar(componentes, solcp.explained_variance_ratio_,
                    label="Explicada", color="#1f77b4")
    ax.plot(componentes, np.cumsum(solcp.explained_variance_ratio_),
            marker='o', label="Acumulada", color="#060606")
    # Etiquetas de porcentaje sobre las barras (formato común)
    ax.bar_label(barras, labels=[f"{h*100:.1f}%" for h in solcp.explained_variance_ratio_], padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(componentes)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('% varianza')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc='center right')
    plt.show()


def optim_ncomp(solcp, vexp):
    """
    Devuelve el nº mínimo de componentes para alcanzar una variabilidad explicada dada.

    Parámetros:
    - solcp: objeto PCA ya ajustado
    - vexp: variabilidad explicada requerida, en porcentaje (p. ej. 90)
    """
    cumVar = np.cumsum(solcp.explained_variance_ratio_) * 100
    valor = int(np.sum(cumVar < vexp) + 1)
    return min(valor, solcp.n_components_)


def plot_contrib(loadings, cp, lx, ly):
    """
    Representa los loadings de una componente principal concreta.

    Parámetros:
    - loadings: DataFrame de loadings (con columna 'Variable')
    - cp: nombre de la componente a representar (p. ej. 'CP1')
    - lx, ly: ancho y alto del gráfico
    """
    loadings_ord = loadings.sort_values(by=cp, ascending=False)
    plt.figure(figsize=(lx, ly))
    colors = ['green' if x > 0 else 'red' for x in loadings_ord[cp]]
    barras = plt.barh(loadings_ord['Variable'], loadings_ord[cp], color=colors)
    plt.gca().bar_label(barras, fmt='%.2f', padding=3)
    plt.xlabel('Contribución')
    plt.ylabel('Variable')
    plt.title(f'Contribución de las variables en {cp}')
    plt.show()


def biplot_loadings(loadings, hg, wd):
    """
    Representa los loadings de las dos primeras componentes principales.

    Parámetros:
    - loadings: DataFrame de loadings (con columna 'Variable')
    - hg: alto (height) del gráfico
    - wd: aspecto (aspect) del gráfico
    """
    g = sns.relplot(data=loadings, x="CP1", y="CP2", height=hg, aspect=wd, color="black")
    ax = g.ax
    ax.axvline(x=0, color='r', linestyle='dotted')
    ax.axhline(y=0, color='r', linestyle='dotted')
    ax.set_xlabel("Componente 1"); ax.set_ylabel("Componente 2")
    for i in range(len(loadings)):
        ax.arrow(0, 0, loadings.CP1.iloc[i], loadings.CP2.iloc[i], color='black')
        ax.scatter(loadings.CP1.iloc[i], loadings.CP2.iloc[i], color='black')
        ax.text(loadings.CP1.iloc[i] + 0.01, loadings.CP2.iloc[i], loadings.Variable.iloc[i],
                ha='left', va='center')
    plt.show()


def _paleta_target(y):
    """Devuelve la paleta adecuada según el tipo de la variable objetivo."""
    if y.dtype == 'object' or y.dtype.name in ('category', 'boolean'):
        return sns.color_palette("hls", len(np.unique(np.asarray(y))))
    return sns.color_palette("coolwarm", as_cmap=True)


def biplot_coordenadas(projected, explained_variance_ratio, y=None, hg=6, wd=1.3):
    """
    Representa las coordenadas (scores) de las muestras en las dos primeras componentes,
    coloreadas opcionalmente según la variable objetivo.

    Parámetros:
    - projected: DataFrame de coordenadas de cada muestra en las CP (scores)
    - explained_variance_ratio: Array o lista con la proporción de varianza explicada por cada componente.
    - y: (Opcional) Serie o array-like con la variable objetivo (etiquetas de grupo).
         Si es None, los puntos no se colorearán por grupos.
    - hg: Alto (height) del gráfico.
    - wd: Aspecto (aspect) del gráfico (relación ancho/alto).
    """
    sns.relplot(x='CP1', y='CP2', hue=y, data=projected, s=60,
                height=hg, aspect=wd, palette='tab10', legend='full')
    plt.title('Gráfico de Coordenadas de las Muestras')
    # Usamos la explained_variance_ratio del PCA para las etiquetas de los ejes
    plt.xlabel(f'CP1 ({round(explained_variance_ratio[0] * 100, 2)}% varianza)')
    plt.ylabel(f'CP2 ({round(explained_variance_ratio[1] * 100, 2)}% varianza)')
    plt.axhline(0, color='gray', lw=.5)
    plt.axvline(0, color='gray', lw=.5)
    plt.show()


def biplot_conjoint(loadings, projected, y, hg, wd):
    """
    Biplot: combina coordenadas (scores) de las muestras y loadings de las variables
    en las dos primeras componentes.

    Parámetros:
    - loadings: DataFrame de loadings (con columna 'Variable')
    - projected: DataFrame de coordenadas de cada muestra en las CP
    - y: variable objetivo (Series)
    - hg: alto (height) del gráfico
    - wd: aspecto (aspect) del gráfico
    """
    datos = projected.copy()
    datos['Target'] = np.asarray(y)
    # Escalamos los scores para que convivan con los loadings
    scalex = 1.0 / (datos["CP1"].max() - datos["CP1"].min())
    scaley = 1.0 / (datos["CP2"].max() - datos["CP2"].min())
    datos["CP1e"] = datos["CP1"] * scalex
    datos["CP2e"] = datos["CP2"] * scaley

    g = sns.relplot(data=datos, x="CP1e", y="CP2e", hue="Target",
                    palette=_paleta_target(y), height=hg, aspect=wd)
    ax = g.ax
    for i in range(len(loadings)):
        ax.arrow(0, 0, loadings.CP1.iloc[i], loadings.CP2.iloc[i], color='black')
        ax.scatter(loadings.CP1.iloc[i], loadings.CP2.iloc[i], color='black')
        ax.text(loadings.CP1.iloc[i] + 0.01, loadings.CP2.iloc[i], loadings.Variable.iloc[i],
                ha='left', va='center')
    ax.axvline(x=0, color='r', linestyle='dotted')
    ax.axhline(y=0, color='r', linestyle='dotted')
    ax.set_xlabel("Componente 1"); ax.set_ylabel("Componente 2")
    plt.show()


def graf_PCA(mpca, figura=(12, 4.5)):
  """
  Función para obtener el gráfico de sedimentación, el de varianza acumualda, y la tabla de varianza acumulada
  para un modelo PCA.

  Argumentos entrada:
  - mpca: modelo PCA a evaluar
  - figura: tamaño de la figura a generar

  Devuelve:
  - gráfico sedimentación
  - gráfico varianza acumulada
  - tabla varianza acumulada
  """
  ratio = mpca.explained_variance_ratio_        # proporción explicada por cada componente
  acum = np.cumsum(ratio)
  comp = np.arange(1, len(ratio) + 1)
  # Diseño de gráfico
  fig, ax = plt.subplots(1, 2, figsize=figura)
  # Gráfico de sedimentación
  ax[0].plot(comp, ratio, 'o-')
  ax[0].set_title('Gráfico de sedimentación (scree)')
  ax[0].set_xlabel('nº de componentes')
  ax[0].set_ylabel('Varianza explicada')
  # Gráfico de varianza acumualda
  ax[1].plot(comp, acum, 'o-')
  ax[1].axhline(0.70, ls='--', color='red')      # umbral del 70 %
  ax[1].axhline(0.80, ls='--', color='orange')      # umbral del 80 %
  ax[1].axhline(0.90, ls='--', color='green')      # umbral del 70 %
  ax[1].set_title('Varianza explicada acumulada')
  ax[1].set_xlabel('nº de componentes')
  ax[1].set_ylabel('Proporción acumulada')
  ax[1].set_ylim(0, 1.02)
  plt.tight_layout()
  plt.show()
  # Tabla de varianza acumulada
  varexp = 100*ratio
  cumvar = pd.DataFrame(varexp.cumsum(), columns=["Varianza acumulada"]) # Convert cumvar to DataFrame
  expvar = pd.DataFrame(varexp, columns=["Varianza explicada"])
  pd.options.display.float_format = '{:.4f}'.format
  tabla = pd.concat([expvar, cumvar], axis=1).set_index(comp)
  return tabla 
    
