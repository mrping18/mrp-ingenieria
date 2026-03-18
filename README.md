# MRP Ingeniería

Aplicación web enfocada en analítica y optimización de datos productivos (Data Science + Procesos) diseñada específicamente para cruzar información de variables de polvo atomizado vs. defectos de planta.

## Arquitectura

- **Backend**: Python 3, Flask.
- **Frontend**: HTML5, Bootstrap 5, Plotly.js para gráficos interactivos.
- **Data & Math**: Pandas, Numpy, Statsmodels (OLS Regresión), SciPy (Optimización L-BFGS-B).
- **Despliegue**: Preparado de forma nativa para Vercel Serverless Functions (`vercel.json` y `api/index.py`).

## Estructura de Carpetas
```
mrp_ingenieria/
├── api/             # Carpeta raíz para Vercel Serverless
│   └── index.py     # La aplicación principal Flask (Controller/Router)
├── templates/       # Vistas HTML (Jinja2)
│   ├── base.html    # Layout principal y diseño (violetas/pastel)
│   ├── index.html   # Pantalla de inicio
│   ├── upload.html  # Interfaz de carga de datos .xlsx
│   ├── analysis.html# Gráficos (Correlación, barras)
│   └── ...
├── requriements.txt # Dependencias pip
├── vercel.json      # Configuración de despliegue para Vercel
└── README.md
```

## Ejecución en Local (Pruebas)

1. **Crear Entorno Virtual**:
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

2. **Instalar Dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar servidor local de desarrollo**:
   ```bash
   # Dentro de la carpeta mrp_ingenieria/
   python api/index.py
   # (Se recomienda añadir bloque if __name__ == '__main__': app.run(debug=True) en api/index.py solo para pruebas locales)
   ```

## Despliegue en Vercel

1. Instala el CLI de Vercel en tu computadora (`npm i -g vercel`).
2. Abre la terminal en la carpeta `mrp_ingenieria`.
3. Ejecuta el comando:
   ```bash
   vercel
   ```
4. Sigue las instrucciones en consola. Vercel leerá automáticamente el archivo `vercel.json` que indica que `api/index.py` contendrá la aplicación.

## Capacidades

- **Cruce de Datos**: Acepta un libro de Excel (ej. "Seguimiento Polvo Atomizado.xlsx") y cruza automáticamente los promedios de la pestaña de variables con las sumas de la pestaña de defectos.
- **Análisis Estadístico Avanzado**: Genera sobre la marcha una matriz de correlaciones numéricas de Pearson y las visualiza en un Heatmap inteligente y en un gráfico de barras unidireccional para identificar los causantes reales de roturas y defectos de prensa.
- **Múltiples Enfoques de Simulación**:
  1. Integración de `Scipy` para encontrar el punto en el Hyper-Plano N-dimensional (Modelo OLS Lineal) que minimiza los defectos a casi cero.
  2. Generación programática de la matriz de **Solver para Microsoft Excel**. Permite bajar un .xlsx que el ingeniero puede abrir, darle clic a Solver en su PC y ver la misma matemática en su herramienta de confort.
