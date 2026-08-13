# 🧪 DREAM ML: Sistema de Gestión de Experimentos en Aprendizaje Automático

> Proyecto académico centrado en mejorar la reproducibilidad y trazabilidad en entornos académicos mediante la integración de herramientas MLOps como MLflow y DVC.

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

---

## 📌 Descripción

DREAM ML es un sistema modular que automatiza y documenta el ciclo de vida de experimentos de *machine learning*, orientado principalmente a modelos de clasificación.  
Se enfoca en facilitar la replicación, seguimiento y versionado de datos, modelos y métricas, utilizando herramientas de código abierto.

---

## 🚀 Características principales

- 🔁 Automatización del ciclo de vida de experimentos  
- 📦 Versionado de datos y modelos con **DVC**  
- 🧾 Registro y documentación automática con **MLflow**  
- 📊 Visualización de métricas y artefactos  
- 🧪 Comparación de modelos  
- ⚡ Medición del consumo energético con **CodeCarbon**  
- 📤 Generación de reportes automáticos  

---

## ⚙️ Tecnologías utilizadas

| Categoría           | Herramientas                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| **Lenguajes**       | Python, JavaScript                                                            |
| **Backend**         | Django                                                                        |
| **Frontend**        | React                                                                         |
| **Bases de datos**  | SQLite                                                                         |
| **Tracking & Charts** | MLflow, Matplotlib, Seaborn                                                |
| **Versionado datos**| DVC                                                                           |
| **Infraestructura** | Docker, Docker Compose                                                        |
| **EDA y Feature Engineering** | Pandas, Numpy, Sweetviz, YData Profiling                            |
| **Experimentación** | Scikit-learn, XGBoost                                                         |
| **Control de versiones** | Git                                                                      |
| **Consumo energético** | CodeCarbon                                                                |

---
## 🧪 Proceso de testing

Durante el desarrollo se realizaron pruebas funcionales para validar:
- Correcto versionado de datasets y modelos con DVC
- Registro automático de experimentos en MLFlow
- Reentrenamiento reproducible
- Reportes automáticos generados correctamente
- Visualización de métricas
- Validación de trazabilidad con diferentes datasets

## 📂 Documentos presentes

- Manual de usuario del sistema
- Diagrama de flujo del sistema
- Diagrama de casos de uso del sistema
- Checklist de cumplimiento de reproducibilidad basado en machine learning reproducibility checklist desarrollado por Joelle Pineau
- Test realizado para validar reproducibilidad

## 🛠️ Instalación y ejecución

### 🔧 Requisitos

- Python 3.10+  
- Docker y Docker Compose  
- Git


### 🚀 Levantar entorno local
- Configura el archivo .env para indicar la carpeta en donde se guardarán los experimentos ejecutados en tu equipo.
```bash
git clone https://github.com/lespinozaortiz/GEML
cd GEML
docker-compose up --build
```
> Este software está licenciado bajo la Licencia Pública General GNU v3.0 (GPLv3).  

## Development workflow

  Most features were built from approved designs with **Claude Code** (VS Code extension and CLI) as the primary development
  environment, in **Interactive and Plan modes**, following a developer-controlled loop: spec → research → plan →
  implement, with generated code reviewed and validated before merging.

  Contributed approximately 50% of the codebase (~5–10 KLOC).
