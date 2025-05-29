import numpy as np          
import pandas as pd         
import math
import random                 
import matplotlib.pyplot as plt
import seaborn as sns

def var_contrib(solcp, X):
    """
    Función para obtener los loadings de cada variable en el análisis de CP lineales, y las coordenadas de los sujetos en las CP

    Parámetros de entrada:
    - solcp: dataframe con las componentes principales
    - X: dataframe de datos utilzado para la solución CP

    Devuelve:
    - Dataframe de pesos con nombres de variables
    - Dataframe de coordenadas de los sujetos en las CP
    """
    # Obtener el número de componentes extraídas por PCA
    n_components = solcp.n_components_
    # Crear los nombres de las componentes
    component_names = [f"CP{i+1}" for i in range(n_components)]

    # Crear DataFrame de loadings
    # Las filas son las variables originales, las columnas son las componentes
    pesos = pd.DataFrame(solcp.components_.T, index=X.columns, columns=component_names)

    # Crear DataFrame de coordenadas
    # Las filas son las muestras, las columnas son las componentes
    coordenadas = pd.DataFrame(solcp.transform(X), index=X.index, columns=component_names)

    return pesos, coordenadas


def plot_var_explained(solcp, lx, ly):
    """
    Función para obtenr la tabla  y representación gráfica de la proporción de varianza explicada por cada componente principal

    Parámetros de entrada:
    - solcp: dataframe con la solución de las componentes principales
    - lx: anchura del eje x
    - ly: anchura del eje y
    """    
    componentes = np.arange(solcp.n_components_) + 1
    cumVar = pd.DataFrame(np.cumsum(solcp.explained_variance_ratio_)*100,
                      columns=["Varianza acumulada"])
    varexp = 100*solcp.explained_variance_ratio_
    expVar = pd.DataFrame(varexp, columns=["Varianza explicada"])
    pd.options.display.float_format = '{:.4f}'.format
    tabla = pd.concat([expVar, cumVar], axis=1).set_index(componentes)
    ## Tabla
    print("Tabla de variablidad explicada")
    print("==============================")
    print(tabla)
    ## Gráfico
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(lx, ly))
    # Varianza explicada por cada componente
    ax.bar(
            x      = np.arange(solcp.n_components_) + 1,
            height = solcp.explained_variance_ratio_,
            label = "Explicada",
            color = "#1f77b4"
    )
    # varianza acumulada
    ax.plot(np.arange(solcp.n_components_) + 1, np.cumsum(solcp.explained_variance_ratio_),
            label="Acumulada", color="#060606")
    # Configuraciones y leyendas
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(1, solcp.n_components_+ 1, 1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('% varianza')

    # Format y-axis ticks as percentages
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(loc='center right')
    plt.show()


def optim_ncomp(solcp, vexp):
    """
    Función para obtenr la tabla  y representación gráfica de la proporción de varianza explicada por cada componente principal

    Parámetros de entrada:
    - solcp: dataframe con la solución de las componentes principales
    - vexp: variabildiad explicada requerida expresada en porcentaje

    """
    componentes = np.arange(solcp.n_components_) + 1
    cumVar = pd.DataFrame(np.cumsum(solcp.explained_variance_ratio_)*100,
                      columns=["Vacum"])
    varexp = 100*solcp.explained_variance_ratio_
    expVar = pd.DataFrame(varexp, columns=["Vexp"])
    pd.options.display.float_format = '{:.4f}'.format
    tabla = pd.concat([expVar, cumVar], axis=1).set_index(componentes)
    valor = sum(tabla["Vacum"] < vexp) +1
    return valor

def plot_contrib(loadings, cp, lx, ly):
  """
  Función para representar los loadings de una componente principal específica

  Parámetros de entrada:
  - loadings: dataframe con los loadings de cada variable en cada componente principales
  - cp: componente principal a representar
  - lx: anchura del eje x (contribuciones)
  - ly: anchura del eje y (variables)
  """
  loadings_ord = loadings.sort_values(by=cp, ascending=False)
  # Crear el gráfico de barras
  plt.figure(figsize=(lx, ly))
  colors = ['green' if x > 0 else 'red' for x in loadings_ord[cp]]
  plt.barh(loadings_ord['Variable'], loadings_ord[cp], color=colors)
  plt.xlabel('')
  plt.ylabel('Variable')
  plt.title('Contribución')
  plt.show()

def biplot_loadings(loadings, hg, wd):
  """
  Función para representar los loadings de las dos primeras componentes principales

  Parámetros de entrada:
  - loadings: dataframe con los loadings de cada variable en cada componente principales
  - hg: height del gráfico
  - wd: aspecto del gráfico

  """  
  sns.relplot(x=loadings["CP1"], y=loadings["CP2"], data=loadings, height = hg, aspect = wd, color="black")
  plt.axvline(x = 0, color = 'r', linestyle = 'dotted')
  plt.axhline(y = 0, color = 'r', linestyle = 'dotted')
  plt.xlabel("Componente 1")
  plt.ylabel("Componente 2")
  for i in range(loadings.shape[0]):
        plt.text(loadings.CP1[i]+0.01, loadings.CP2[i], loadings.Variable[i],ha = 'left', va = 'center')
  for i in range(len(loadings)):
    plt.arrow(0, 0, loadings['CP1'][i], loadings['CP2'][i], color='black')
    plt.scatter(loadings.CP1[i], loadings.CP2[i], color = 'black')
  plt.show()


def biplot_coordenadas(projected, y, hg, wd):
  """
  Función para representar los loadings de las dos primeras componentes principales

  Parámetros de entrada:
  - projected: dataframe con los loadings de cada variable en cada componente principales
  - y: variable objetivo
  - hg: height del gráfico
  - wd: aspecto del gráfico

  """
  # Combinamos coordendas con target
  projected =pd.concat([y, projected], axis=1).rename({y.name: 'Target'},axis=1)
  if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype.name == 'boolean':
        sns.relplot(x = projected["CP1"], y = projected["CP2"], hue = projected["Target"],
            palette= sns.color_palette("hls", len(np.unique(projected["Target"]))),
            data=projected, height = hg, aspect = wd)
  else:
        sns.relplot(x = projected["CP1"], y = projected["CP2"], hue = projected["Target"],
            palette= sns.color_palette("coolwarm", as_cmap=True),
            data=projected, height = hg, aspect = wd)

  plt.axvline(x = 0, color = 'r', linestyle = 'dotted')
  plt.axhline(y = 0, color = 'r', linestyle = 'dotted')
  plt.xlabel("Componente 1")
  plt.ylabel("Componente 2")
  plt.show()


def biplot_conjoint(loadings, projected, y, hg, wd):
  """
  Función para representar los loadings y coordenadas de las dos primeras componentes principales

  Parámetros de entrada:
  - loadings: dataframe con los loadings de cada variable en cada componente principal
  - projected: dataframe con las coordenadas de cada muestra en cada componente principal
  - y: variable objetivo
  - hg: height del gráfico
  - wd: aspecto del gráfico

  """
  # Combinamos coordendas con target
  projected = pd.concat([y, projected], axis=1).rename({y.name: 'Target'},axis=1)
  xs = projected["CP1"]
  ys = projected["CP2"]
  n = loadings.shape[0]
  # Escalamos scores para ajustar a loadings
  scalex = 1.0/(xs.max() - xs.min())
  scaley = 1.0/(ys.max() - ys.min())
  ### gráfico de coordenadas
  if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype.name == 'boolean':
        sns.relplot(x = projected["CP1"]*scalex, y = projected["CP2"]*scaley, hue = projected["Target"],
            palette= sns.color_palette("hls", len(np.unique(projected["Target"]))),
            data=projected, height = hg, aspect = wd)
  else:
        sns.relplot(x = projected["CP1"]*scalex, y = projected["CP2"]*scaley, hue = projected["Target"],
            palette= sns.color_palette("coolwarm", as_cmap=True),
            data=projected, height = hg, aspect = wd)
  ### gráfico de loadings
  for i in range(loadings.shape[0]):
        plt.text(loadings.CP1[i]+0.01, loadings.CP2[i], loadings.Variable[i], ha = 'left', va = 'center')
  for i in range(len(loadings)):
    plt.arrow(0, 0, loadings['CP1'][i], loadings['CP2'][i], color='black')
    plt.scatter(loadings.CP1[i], loadings.CP2[i], color = 'black')

  plt.axvline(x = 0, color = 'r', linestyle = 'dotted')
  plt.axhline(y = 0, color = 'r', linestyle = 'dotted')
  plt.xlabel("Componente 1")
  plt.ylabel("Componente 2")
  plt.show()
