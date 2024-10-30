# Proyecto de Machine Learning: Análisis y Predicción de Precios de Zapatillas

## Descripción del Proyecto

Este proyecto realiza un análisis completo de datos y modelos de Machine Learning para predecir precios de las zapatillas utilizando un conjunto de datos de ventas de StockX. 

## Objetivos del Proyecto

- Analizar y visualizar los datos de ventas de zapatillas.
- Desarrollar modelos predictivos para estimar el precio de venta de los modelos de zapatillas.
- Evaluar los márgenes de beneficio y recomendar estrategias de venta.
- Desplegar una interfaz de usuario interactiva para la predicción de precios.

## Pasos Realizados

### Definición del Problema
Definimos el problema de predicción de precios y rentabilidad en el mercado de zapatillas.

### Obtención de Datos
Utilizamos un conjunto de datos de StockX que incluye aproximadamente 100,000 transacciones de zapatillas. El dataset contiene información sobre el modelo, fecha de venta, precio de venta, talla y precio original.

### Métricas de Evaluación
Se definieron las métricas de evaluación, como el **Raíz del Error Cuadrático Medio (RMSE)** , el  **Error cuadrático medio (MSE)** , el **Error Absoluto Medio (MAE)** y el **Coeficiente de Determinación (R<sup>2</sup>)**. Además, se separaron los datos en conjuntos de **entrenamiento** , **prueba** y **validación** para validar la efectividad de los modelos.



### Limpieza y Preprocesamiento de Datos

Realizamos una limpieza y preprocesamiento de datos detallado para estructurar y normalizar la información del dataset de sneakers. A continuación, se describen los pasos clave:

- **Normalización de Marcas y Colaboraciones**: 
  - Estandarizamos los nombres de las marcas y colaboraciones (por ejemplo, “adidas” se normalizó a “Adidas”).
  - Identificamos sub-marcas y asignamos marcas principales, como “Yeezy” a “Adidas” y “Air Jordan” a “Nike”.

- **Asignación de Tokens de Productos**:
  - Dividimos los nombres de cada sneaker en tokens para extraer información específica, como el **Producto**, **Marca**, **Colaboración**, **Modelo**, **Color** y **Año de Lanzamiento**.
  - Clasificamos estos elementos en categorías predefinidas y asignamos sus valores a cada sneaker.

- **Formateo de Precios y Fechas**:
  - Convertimos los valores de “Precio de Venta” y “Precio de Retail” de strings con formato de moneda a valores numéricos.
  - Estandarizamos los campos de fechas a un formato de timestamp para facilitar el análisis temporal.

- **Eliminación de Columnas Irrelevantes**:
  - Se eliminaron las siguientes columnas que no eran necesarias para el análisis:
    - **"Sneaker Name"**: Nombre del sneaker original, ya que se normalizó y se transformó.
    - **"Buyer Region"**: Información sobre la región del comprador, que no se consideró relevante para las predicciones.
    - **"Year"**: Esta columna, derivada del nombre del sneaker, contenía un 90% de valores nulos así que se considero no usarla.

- **Transformación de Características Categóricas**:
  - Realizamos un proceso de one-hot encoding en variables categóricas como **Marca**, **Sub-marca**, **Modelo** y **Colaboración**.
  - Codificamos los colores de cada sneaker, creando columnas binarias para cada color.

- **Creación de Variables Temporales**:
  - Generamos variables derivadas de las fechas de compra y lanzamiento, como el año, mes, día y día de la semana para cada evento.

- **Estandarización de Características Numéricas**:
  - Utilizamos `StandardScaler` para normalizar los valores de características numéricas, como “Precio de Retail” y “Talla del Zapato”.
  - Guardamos el modelo de escalado con `pickle` para garantizar la coherencia en el procesamiento de nuevos datos.

Este proceso detallado de limpieza y preprocesamiento garantiza que los datos estén estructurados y preparados para el análisis y el entrenamiento de modelos de predicción con alta precisión.



### Entrenamiento y Selección del Modelo

Se probaron y entrenaron varios modelos de Machine Learning y se evaluaron en función de las métricas **RMSLE**, **RMSE**, **MAE**, **R²** y el tiempo promedio de predicción:

- **RandomForestRegressor (RFR)**: Este modelo obtuvo el mejor rendimiento en todas las métricas de precisión, con un RMSLE de 0.0465, RMSE de 35.05 y el valor más alto de R² (0.9809). Sin embargo, el tiempo promedio de predicción fue de 21.63 segundos, lo cual no es ideal para aplicaciones que requieren rapidez.

- **XGBRegressor (XGBR)**: Se destacó por su balance entre precisión y eficiencia, logrando un RMSLE de 0.0519, RMSE de 36.23 y un tiempo promedio de predicción de solo 0.37 segundos. Este modelo ofreció un desempeño robusto y tiempos de respuesta mucho más rápidos, ideales para su implementación en la aplicación.

- **CatBoostRegressor (CAT)**: Con un RMSLE de 0.0539 y RMSE de 36.96, mostró un rendimiento sólido, aunque con tiempos de predicción ligeramente más altos (6.06 segundos en promedio).

- **HistGradientBoostingRegressor (HistGBR)** y **LightGBM (LGBM)**: Ambos modelos presentaron RMSLEs alrededor de 0.066, con RMSEs en torno a 43.8. Aunque son más rápidos que RFR, fueron superados por XGBR en precisión y eficiencia.

- Otros modelos, como **KNeighborsRegressor (KNN)**, **GradientBoostingRegressor (GBR)** y **LinearRegression (LR)**, mostraron desempeños inferiores en términos de precisión y tiempos de predicción.

**Selección del Modelo**:
Finalmente, se seleccionó **XGBRegressor** para la aplicación debido a su óptimo balance entre precisión y tiempo de respuesta (0.37 segundos en promedio), cumpliendo con los requisitos de rapidez y desempeño para el despliegue.



### Test y Análisis de Resultados

Después de entrenar y probar varios modelos, seleccionamos **XGBoost** debido a su excelente desempeño en las métricas de evaluación, su tiempo de ejecución bajo y su eficiencia en términos de uso de recursos.

**Resultados de XGBoost**:

- **Entrenamiento**:
  - RMSLE: 0.0446
  - RMSE: 25.72
  - MAE: 15.25
  - R²: 0.9900

- **Validación**:
  - RMSLE: 0.0519
  - RMSE: 36.23
  - MAE: 17.86
  - R²: 0.9795

- **Prueba**:
  - RMSLE: 0.0519
  - RMSE: 36.21
  - MAE: 17.80
  - R²: 0.9797

Estos resultados muestran la capacidad de **XGBoost** para mantener un alto nivel de precisión y un rendimiento consistente en todas las fases de evaluación. Su velocidad de predicción también lo convierte en el modelo óptimo para nuestras necesidades de predicción de precios.


### Interpretación
Se interpretaron los resultados obtenidos, destacando la importancia de factores como el modelo, la talla y la fecha de lanzamiento
 en la predicción de precios.

### Despliegue
El modelo se implementó en producción utilizando **Gradio** para crear una interfaz de usuario interactiva, y **Docker** para contenerizar la aplicación y facilitar su despliegue en diferentes entornos. Puedes acceder a la aplicación en [http://localhost:7860](http://localhost:7860) una vez que se ejecute el contenedor.


## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal.
- **Pandas**: Para manipulación y análisis de datos.
- **Matplotlib y Seaborn**: Para visualización de datos.
- **Sweetviz**: Para analizar los datos y verificar la distribución equitativa del dataset al dividirlo.
- **Pickle**: Para guardar el modelo y el scaler.
- **Scikit-learn**: Para creación y evaluación de modelos de Machine Learning.
- **XGBoost**: Modelo usado en la aplicación
- **LightGBM**: Modelo usado para comparar
- **CatBoost**: Modelo usado para comparar
- **Gradio**: Para crear una interfaz de usuario interactiva.
- **Docker**: Para contenerización y despliegue de la aplicación.



## Instalación
