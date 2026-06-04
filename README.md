# Aplicacion de Ciencia de Datos - Streamlit

Este proyecto consiste en una aplicacion interactiva de ciencia de datos desarrollada con Streamlit. La aplicacion integra herramientas de analisis exploratorio de datos, entrenamiento de modelos de aprendizaje automatico, un sistema de recomendacion, analisis de archivos cargados por el usuario, scraping y analisis de sentimientos local, y una interfaz de preguntas basada en procesamiento de lenguaje natural basico.

El diseño del sitio es limpio, visualmente sofisticado y no incluye ningun emoticón o emoji de acuerdo con los requerimientos establecidos.

---

## Requisitos Previos

Asegurese de tener instalado Python en su sistema (version 3.9 o superior recomendada). Las dependencias necesarias son:

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl
- beautifulsoup4
- requests

---

## Instalacion y Configuracion

Siga estos pasos para configurar el entorno y ejecutar la aplicacion:

1. Instale las librerias requeridas ejecutando el siguiente comando en la terminal o linea de comandos:
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: Si no cuenta con el archivo de requerimientos todavia, puede instalar las librerias una a una:*
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl beautifulsoup4 requests
   ```

2. Verifique que los conjuntos de datos se encuentren en la ruta correcta dentro de su carpeta de trabajo:
   - `data/imdb_top_movies_1980_2026.csv`
   - `data/retail_fraud_detection_100k.csv`

---

## Ejecucion de la Aplicacion

Para iniciar el servidor local de Streamlit y abrir la aplicacion en su navegador web:

1. Abra una terminal en el directorio raiz del proyecto (donde se encuentra `Inicio.py`).
2. Ejecute el siguiente comando:
   ```bash
   streamlit run Inicio.py
   ```
3. La aplicacion se abrira automaticamente en su navegador web predeterminado en la direccion local (por lo general, http://localhost:8501).

---

## Estructura de Opciones en el Sitio

La aplicacion se organiza mediante una barra de navegacion en el panel lateral que divide el trabajo de la siguiente forma:

- **Opcion 1: Inicio**: Pagina de presentacion que explica la estructura general del proyecto.
- **Opcion 2: Analisis Exploratorio**:
  - **Descripcion del dataset**: Informacion basica y estructura del dataset seleccionado.
  - **Descripcion de los campos**: Resumen estadistico interactivo. Muestra estadisticas detalladas para variables numericas y valores unicos con conteo para variables categoricas.
  - **Navegador del dataset completo**: Tabla interactiva con opciones para ordenar y filtrar registros.
  - **Buscador de registros**: Localizacion rapida de filas mediante identificadores unicos (imdb_id o transaction_id).
  - **Graficador exploratorio**: Generador automatico de diagramas de barras, histogramas o diagramas de caja dependiendo de las caracteristicas de la columna.
  - **Hipotesis**: Evaluacion estadistica y grafica de hipotesis planteadas sobre el dataset seleccionado.
- **Opcion 3: Aprendizaje Automático**:
  - Permite configurar y entrenar modelos predictivos ajustando el porcentaje de entrenamiento/prueba, seleccionando variables y comparando algoritmos (como Regresiones y Arboles de Decision). Se reportan metricas de precision mediante componentes st.metric y graficos comparativos.
- **Opcion 4: Sistema de Recomendacion**:
  - Motor de recomendacion de videojuegos que calcula la similitud entre titulos del catalogo y sugiere opciones personalizadas basadas en preferencias de genero, plataformas y descripcion.
- **Opcion 5: Analisis de Datos por Carga de Archivos**:
  - Seccion abierta para subir cualquier archivo local en formato CSV o Excel para visualizar su estructura y generar representaciones graficas dinamicas al instante.
- **Opcion 6: Analisis de Sentimientos y Scrapping**:
  - Extrae reseñas de un foro local simulado en HTML y realiza una clasificacion automatica del sentimiento (Positivo, Neutro o Negativo) reportando la polaridad promedio.
- **Opcion 7: Prompts de IA**:
  - Interfaz interactiva que interpreta consultas en lenguaje natural del usuario sobre el dataset y responde con precision sobre promedios, maximos, minimos y dimensiones de la tabla.

---

## Indicaciones

Elabore una aplicación Streamlit que incluya los siguientes elementos en una barra de navegación:

**Opción 1: Inicio**

**Opción 2: Análisis Exploratorio:** Dividirlo en los siguientes submenús:
- Descripción del dataset.
- Descripción de los campos: Que el usuario seleccione un campo y pueda ver su descripción y las medidas de describe (en el caso de que sean campos cuantitativos) y en el caso de los campos categóricos, cuáles son los posibles valores.
- Navegador del dataset completo.
- Buscador de registros en base a un código. (No obligatorio – Bonus).
- Graficador exploratorio: El usuario selecciona un campo de la tabla y se le muestra el gráfico adecuado para ello.
- Hipótesis: Incluir por lo menos dos hipótesis. El usuario selecciona una hipótesis y le aparece el análisis no gráfico o gráfico que valida o no valida la hipótesis. Al final se incluye una conclusión del análisis.

**Opción 3: Aprendizaje Automático:** Elaborar una interfaz que permita al usuario:
- Seleccionar el algoritmo (incluir por lo menos dos)
- Seleccionar una variable a analizar (incluir por lo menos dos)
- Seleccionar una variable independiente (incluir por lo menos dos)
- Que permita seleccionar la cantidad de datos de entrenamiento y de prueba para el proyecto (en porcentaje).
- Que muestre en una gráfica el efecto de estos cambios. Incluir en la gráfica tanto los datos de prueba (predicciones) como los datos de entrenamiento.
- Mostrar los parámetros de entrenamiento (coeficientes, accuracy, etc) utilizando st.metric de Streamlit u otro control.

**Opción 4: Sistema de recomendación:** Elegir uno de los siguientes sistemas de recomendación e implementarlo
- Música
- Libros y novelas digitales
- Cursos en línea
- Empleos
- Amigos o conexiones
- Restaurantes
- Destinos turísticos
- Videojuegos
- Productos en tienda en línea
- Noticias
- Podcast
- Ropa
- Hoteles o alojamientos
- Eventos o conciertos
- Recetas de cocina

*Ingeniería en Sistemas y Redes Informáticas*
*Ciclo I-2026*
*Técnica Electiva I - Ciencia de Datos*
*Tarea Laboratorio I y Laboratorio II – Cómputo 3*

**Opción 5: Análisis de Datos por Carga de Archivos**
- Cree una aplicación que cargue un archivo CSV, Excel o de otro tipo de fuente de datos de la computadora, a través de un navegador de archivos y genere por lo menos un gráfico de los datos del archivo

**Opción 6: Análisis de Sentimientos y Scrapping**
Aplicación que lea opiniones de uno de los siguientes sitios:
- Un foro
- Una página de noticias
- Una publicación de redes sociales
- Otro sitio que exprese opiniones
Luego muestre las opiniones leídas y haga un análisis de sentimientos de dichas opiniones.

**Opción 7 (Opcional): Prompts de IA**
Cree una aplicación que permita recibir Prompts de consulta de datos y dé una respuesta utilizando una interfaz IA.
Ejemplos de preguntas:
¿Cuántas columnas tiene el dataset?
¿Cuál es la media del campo edad?
¿Cuál es el mayor valor del campo calificación?

---

## Rúbrica

- **Opción 1: Inicio**
- **Opción 2: Análisis Exploratorio:** Dividirlo en los siguientes submenús (30%)
  - Descripción del dataset.
  - Descripción de los campos: Que el usuario seleccione un campo y pueda ver su descripción y las medidas de describe (en el caso de que sean campos cuantitativos) y en el caso de los campos categóricos, cuáles son los posibles valores.
  - Navegador del dataset completo.
  - Buscador de registros en base a un código. (No obligatorio – Bonus).
  - Graficador exploratorio: El usuario selecciona un campo de la tabla y se le muestra el gráfico adecuado para ello.
  - Hipótesis: Incluir por lo menos dos hipótesis. El usuario selecciona una hipótesis y le aparece el análisis no gráfico o gráfico que valida o no valida la hipótesis. Al final se incluye una conclusión del análisis.
- **Aprendizaje Automático:** Elaborar una interfaz que permita al usuario (30%)
  - Seleccionar el algoritmo (incluir por lo menos dos)
  - Seleccionar una variable a analizar (incluir por lo menos dos)
  - Seleccionar una variable independiente (incluir por lo menos dos)
  - Que permita seleccionar la cantidad de datos de entrenamiento y de prueba para el proyecto (en porcentaje).
  - Que muestre en una gráfica el efecto de estos cambios. Incluir en la gráfica tanto los datos de prueba (predicciones) como los datos de entrenamiento.
  - Mostrar los parámetros de entrenamiento (coeficientes, accuracy, etc) utilizando st.metric de Streamlit u otro control.
- **Sistema de recomendación** (30%)
- **Pregunta** (10%)

**TAREA 2**
- **Carga de Archivos del Explorador y Análisis de Datos** (40%)
- **Análisis de Sentimientos y Scrapping** (50%)
- **Pregunta o Interfaz IA** (10%)

**PARCIAL**
(Próximamente)

**Fecha entrega Tareas 1 y 2:** Viernes 5 de Junio.
