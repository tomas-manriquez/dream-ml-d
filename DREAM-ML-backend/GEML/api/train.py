# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

import os
import json
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, callback
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import mlflow
from mlflow import log_param, log_metric
from mlflow.models import infer_signature
from mlflow import MlflowClient
from codecarbon import EmissionsTracker
import numpy as np
from scipy.stats import norm
import psutil
import time

# Optuna for Bayesian hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna import Trial

# ======================
# CONFIGURACIÓN GLOBAL
# ======================
SEED = 42
N_JOBS = 1  # Desactivar paralelismo para determinismo
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

logger = logging.getLogger(__name__)

# Callback
class EnableDeterministic(callback.TrainingCallback):
    """
    DEPRECATED: This callback is no longer used.

    XGBoost 3.1.1 is deterministic by default when using tree_method="hist"
    with a fixed random_state parameter. The "deterministic" parameter that
    this callback attempted to set does not exist in XGBoost 3.1.1.

    This class is kept for backward compatibility with existing imports.
    """
    def after_training(self, model):
        # No-op: determinism is already ensured by random_state parameter
        return model

def set_global_seeds():
    """Fija semillas para todas las librerías relevantes"""
    import numpy as np
    import random
    import tensorflow as tf
    
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

# Inicializar semillas globales
set_global_seeds()

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)

# ======================
# FUNCIONES AUXILIARES
# ======================
def load_and_validate_data(dataset_path, input_features, target_variable):
    """
    Carga y valida el dataset
    Retorna:
        DataFrame con datos validados
    """
    df = pd.read_csv(dataset_path)
    missing_cols = [col for col in input_features + [target_variable] 
                   if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {missing_cols}")
    
    # Verificar valores nulos
    for col in input_features + [target_variable]:
        if df[col].isnull().any():
            logger.warning(f"Columna {col} contiene valores nulos")
            
    return df

def split_dataset(X, y, split_ratios):
    """
    Divide el dataset en train/val/test estratificados
    Retorna:
        Tupla con conjuntos divididos (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Validar que los ratios sumen ≈1.0
    total_ratio = split_ratios["train"] + split_ratios["val"] + split_ratios["test"]
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Suma de ratios debe ser 1.0, actual: {total_ratio}")
    
    test_size = split_ratios["val"] + split_ratios["test"]
    val_test_ratio = split_ratios["test"] / test_size
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, 
        random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(model, X, y, prefix, problem_type, experiment_dir):
    """
    Evalúa el modelo y genera métricas/gráficos
    Retorna:
        Diccionario con métricas calculadas y rutas de artefactos
    """
    y_pred = model.predict(X)
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, y_pred)
    }
    
    # Inicializar rutas de artefactos
    artifacts = {
        "confusion_matrix": None,
        "roc_curve": None
    }
    
    # Manejar problemas binarios vs multiclase
    average = "binary" if problem_type == "binary" else "macro"
    y_probs = None
    
    # Obtener probabilidades si el modelo las soporta
    try:
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X)
        else:
            logger.warning(f"Modelo {type(model).__name__} no implementa predict_proba - métricas limitadas")
    except Exception as e:
        logger.error(f"Error en predict_proba: {str(e)}")
        y_probs = None
    
    if problem_type == "binary":
        if y_probs is not None:
            y_probs_bin = y_probs[:, 1]
            metrics.update({
                f"{prefix}_f1": f1_score(y, y_pred),
                f"{prefix}_precision": precision_score(y, y_pred),
                f"{prefix}_recall": recall_score(y, y_pred),
                f"{prefix}_roc_auc": roc_auc_score(y, y_probs_bin)
            })
        else:
            metrics.update({
                f"{prefix}_f1": f1_score(y, y_pred),
                f"{prefix}_precision": precision_score(y, y_pred),
                f"{prefix}_recall": recall_score(y, y_pred),
                f"{prefix}_roc_auc": None
            })
    else:
        metrics.update({
            f"{prefix}_f1": f1_score(y, y_pred, average=average),
            f"{prefix}_precision": precision_score(y, y_pred, average=average),
            f"{prefix}_recall": recall_score(y, y_pred, average=average)
        })
        
        # Calcular ROC-AUC para multiclase si hay probabilidades
        if y_probs is not None:
            try:
                metrics[f"{prefix}_roc_auc"] = roc_auc_score(
                    y, 
                    y_probs, 
                    multi_class='ovo', 
                    average='macro'
                )
            except Exception as e:
                logger.error(f"Error calculando ROC-AUC multiclase: {e}")
                metrics[f"{prefix}_roc_auc"] = None
    
    # Generar gráficos y capturar rutas
    artifacts = generate_plots(
        y, y_pred, 
        y_probs[:, 1] if problem_type == "binary" and y_probs is not None else None,
        prefix, problem_type, experiment_dir
    )
    
    return metrics, artifacts

def generate_plots(y_true, y_pred, y_probs, prefix, problem_type, experiment_dir):
    """Genera y guarda gráficos de evaluación, retorna rutas de artefactos"""
    artifacts = {
        "confusion_matrix": None,
        "roc_curve": None
    }
    
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format='d')
    cm_path = os.path.join(experiment_dir, f"confusion_matrix_{prefix}.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, "plots")
    artifacts["confusion_matrix"] = cm_path
    
    # Curva ROC (para binario con probabilidades)
    if problem_type == "binary" and y_probs is not None:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y_true, y_probs):.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {prefix.capitalize()}')
        plt.legend()
        roc_path = os.path.join(experiment_dir, f"roc_curve_{prefix}.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path, "plots")
        artifacts["roc_curve"] = roc_path
    
    # Curva ROC para multiclase
    elif problem_type == "multiclass" and y_probs is not None:
        try:
            # Calcular curva ROC para cada clase
            n_classes = y_probs.shape[1]
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
                roc_auc = roc_auc_score((y_true == i).astype(int), y_probs[:, i])
                ax.plot(fpr, tpr, label=f'Clase {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Curvas ROC por Clase - {prefix.capitalize()}')
            plt.legend()
            roc_path = os.path.join(experiment_dir, f"multiclass_roc_{prefix}.png")
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path, "plots")
            artifacts["roc_curve"] = roc_path
            
        except Exception as e:
            logger.error(f"Error generando ROC multiclase: {str(e)}")
    
    return artifacts

def log_energy_metrics(tracker):
    """Registra métricas de energía y emisiones"""
    energy_kwh = float(tracker._total_energy or 0.0)
    emissions_kg = float(tracker.final_emissions or 0.0)
    mlflow.log_metric("energy_consumed_total_kWh", energy_kwh)
    mlflow.log_metric("carbon_emission_kg", emissions_kg)
    return energy_kwh, emissions_kg

def save_pipeline_config(experiment_dir, config):
    """Guarda la configuración del pipeline en JSON"""
    config_path = os.path.join(experiment_dir, "pipeline_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing_config = json.load(f)
    else:
        existing_config = {"steps": []}

    existing_config["steps"].append(config)
    with open(config_path, "w") as f:
        json.dump(existing_config, f, indent=4)

# ======================
# RANDOM SEARCH HELPER FUNCTIONS
# ======================

def generate_random_logistic_params(random_search_params: dict) -> dict:
    """
    Genera parámetros aleatorios para Logistic Regression basados en los rangos especificados.

    Args:
        random_search_params: Diccionario con rangos de parámetros

    Returns:
        Diccionario con parámetros aleatorios generados
    """
    # Valores por defecto para los rangos
    default_ranges = {
        "C_range": [0.001, 100.0],
        "max_iter_range": [100, 1000],
        "solver_options": ["lbfgs", "liblinear", "saga"],
        "penalty_options": ["l2", "none"]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros aleatorios
    # Usar muestreo log-uniforme para C
    log_min = np.log(ranges["C_range"][0])
    log_max = np.log(ranges["C_range"][1])
    C = float(np.exp(np.random.uniform(log_min, log_max)))

    max_iter = int(np.random.randint(ranges["max_iter_range"][0], ranges["max_iter_range"][1] + 1))
    solver = str(np.random.choice(ranges["solver_options"]))
    penalty = str(np.random.choice(ranges["penalty_options"]))

    # Validar compatibilidad solver-penalty
    if solver == "lbfgs" and penalty not in ["l2", "none"]:
        penalty = "l2"
    elif solver == "liblinear" and penalty == "elasticnet":
        penalty = "l2"

    params = {
        "C": C,
        "max_iter": max_iter,
        "solver": solver,
        "penalty": penalty,
        "random_state": SEED
    }

    return params

def generate_random_mlp_params(random_search_params: dict) -> dict:
    """
    Genera parámetros aleatorios para MLP basados en los rangos especificados.

    Args:
        random_search_params: Diccionario con rangos de parámetros

    Returns:
        Diccionario con parámetros aleatorios generados
    """
    # Valores por defecto para los rangos
    default_ranges = {
        "hidden_layer_sizes_options": [(4,), (10,), (10, 5), (50,), (100,), (100, 50), (100, 50, 10)],
        "activation_options": ["relu", "tanh", "logistic"],
        "solver_options": ["adam", "sgd"],
        "learning_rate_init_range": [0.0001, 0.1],
        "max_iter_range": [200, 500]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros aleatorios
    # Para hidden_layer_sizes, seleccionar aleatoriamente un índice
    hls_options = [tuple(x) if isinstance(x, list) else x for x in ranges["hidden_layer_sizes_options"]]
    hidden_layer_sizes = hls_options[np.random.randint(0, len(hls_options))]
    activation = str(np.random.choice(ranges["activation_options"]))
    solver = str(np.random.choice(ranges["solver_options"]))
    max_iter = int(np.random.randint(ranges["max_iter_range"][0], ranges["max_iter_range"][1] + 1))

    # Usar muestreo log-uniforme para learning_rate_init
    log_min = np.log(ranges["learning_rate_init_range"][0])
    log_max = np.log(ranges["learning_rate_init_range"][1])
    learning_rate_init = float(np.exp(np.random.uniform(log_min, log_max)))

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "random_state": SEED,
        "shuffle": False
    }

    return params

def generate_random_xgboost_params(random_search_params: dict) -> dict:
    """
    Genera parámetros aleatorios para XGBoost basados en los rangos especificados.

    Args:
        random_search_params: Diccionario con rangos de parámetros

    Returns:
        Diccionario con parámetros aleatorios generados
    """
    # Valores por defecto para los rangos
    default_ranges = {
        "n_estimators_range": [50, 500],
        "max_depth_range": [3, 10],
        "learning_rate_range": [0.01, 0.3],
        "subsample_range": [0.5, 1.0],
        "colsample_bytree_range": [0.5, 1.0],
        "gamma_range": [0.0, 5.0],
        "min_child_weight_range": [1, 10],
        "reg_alpha_range": [0.0, 1.0],
        "reg_lambda_range": [0.0, 1.0]
    }

    # Combinar rangos por defecto con los proporcionados por el usuario
    ranges = {**default_ranges, **random_search_params}

    # Generar parámetros aleatorios
    params = {
        "n_estimators": int(np.random.randint(ranges["n_estimators_range"][0], ranges["n_estimators_range"][1] + 1)),
        "max_depth": int(np.random.randint(ranges["max_depth_range"][0], ranges["max_depth_range"][1] + 1)),
        "min_child_weight": int(np.random.randint(ranges["min_child_weight_range"][0], ranges["min_child_weight_range"][1] + 1)),
        "subsample": float(np.random.uniform(ranges["subsample_range"][0], ranges["subsample_range"][1])),
        "colsample_bytree": float(np.random.uniform(ranges["colsample_bytree_range"][0], ranges["colsample_bytree_range"][1])),
        "gamma": float(np.random.uniform(ranges["gamma_range"][0], ranges["gamma_range"][1])),
        "reg_alpha": float(np.random.uniform(ranges["reg_alpha_range"][0], ranges["reg_alpha_range"][1])),
        "reg_lambda": float(np.random.uniform(ranges["reg_lambda_range"][0], ranges["reg_lambda_range"][1])),
        "random_state": SEED
    }

    # Usar muestreo log-uniforme para learning_rate
    log_min = np.log(ranges["learning_rate_range"][0])
    log_max = np.log(ranges["learning_rate_range"][1])
    params["learning_rate"] = float(np.exp(np.random.uniform(log_min, log_max)))

    return params

# ======================
# BAYESIAN SEARCH HELPER FUNCTIONS
# ======================

def convert_frontend_bayesian_params(frontend_params: dict) -> dict:
    """
    Convierte parámetros de búsqueda Bayesiana del formato frontend al formato backend.

    Frontend format:
        { "C": { "type": "real", "distribution": "log-uniform", "low": 0.001, "high": 100.0 } }

    Backend format:
        { "C": { "type": "float", "low": 0.001, "high": 100.0, "log": True } }

    Args:
        frontend_params: Diccionario con parámetros en formato frontend o backend

    Returns:
        Diccionario con parámetros en formato backend compatible con Optuna
    """
    backend_params = {}

    for param_name, config in frontend_params.items():
        if not isinstance(config, dict):
            continue

        backend_config = {}

        # Check if already in backend format (has "log" field or type is already "float"/"int")
        # Backend format has: type="float"/"int"/"categorical" and "log" field for numeric types
        param_type = config.get("type", "float")
        is_backend_format = (
            param_type in ["float", "int", "categorical"] and
            ("log" in config or param_type == "categorical")
        )

        if is_backend_format:
            # Already in backend format, just copy it
            backend_config = config.copy()
        else:
            # Convert from frontend format
            # Convert type: "real" -> "float", "integer" -> "int", "categorical" -> "categorical"
            if param_type == "real":
                backend_config["type"] = "float"
            elif param_type == "integer":
                backend_config["type"] = "int"
            else:
                backend_config["type"] = param_type  # "categorical" stays same

            # For numeric types, convert distribution to log flag
            if backend_config["type"] in ["float", "int"]:
                backend_config["low"] = config.get("low")
                backend_config["high"] = config.get("high")

                # Convert distribution to log flag
                distribution = config.get("distribution", "uniform")
                backend_config["log"] = (distribution == "log-uniform")

                # VALIDATION: Log-uniform requires positive values
                if backend_config["log"]:
                    low_val = backend_config["low"]
                    high_val = backend_config["high"]

                    if low_val is None or high_val is None:
                        raise ValueError(
                            f"Parameter '{param_name}' with log-uniform distribution "
                            f"requires both 'low' and 'high' to be specified"
                        )

                    if low_val <= 0 or high_val <= 0:
                        raise ValueError(
                            f"Parameter '{param_name}' has distribution='log-uniform' but "
                            f"contains non-positive values. Log-uniform requires low > 0 and high > 0. "
                            f"Got low={low_val}, high={high_val}. "
                            f"Suggested: Use positive values like low=0.0001, high=0.1"
                        )

                    if low_val >= high_val:
                        raise ValueError(
                            f"Parameter '{param_name}' has invalid range: low ({low_val}) >= high ({high_val}). "
                            f"Range must satisfy low < high."
                        )

            # For categorical types, copy choices
            if backend_config["type"] == "categorical":
                backend_config["choices"] = config.get("choices", [])

        backend_params[param_name] = backend_config

    return backend_params

#
# ======================
# FUNCIONES DE ENTRENAMIENTO
# ======================
def train_logistic_regression_model(dataset_path, data, experiment_dir):
    """
    Entrena y registra un modelo de regresión logística para clasificación binaria.
    """
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=experiment_dir, save_to_file=False, allow_multiple_runs=True)
    tracker.start()

    # Extraer parámetros
    input_features = data["input_features"]
    target_variable = data["target_variable"]
    hyperparams = data.get("params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "LogisticRegression_Model")
    problem_type = "binary"

    # Soporte para nuevo formato de búsqueda de hiperparámetros con backward compatibility
    use_grid_search = data.get("use_grid_search", False)  # Deprecated, para backward compatibility
    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", None)

    # Si no se especifica hyperparameter_search_strategy, usar use_grid_search para determinar estrategia
    if hyperparameter_search_strategy is None:
        hyperparameter_search_strategy = "grid" if use_grid_search else "none"

    # Validar estrategia
    valid_strategies = ["none", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}. Recibido: {hyperparameter_search_strategy}")

    # Parámetros para random search
    n_random_iterations = data.get("n_random_iterations", 50)
    random_search_params = data.get("random_search_params", {})

    # Validar parámetros de random search
    if hyperparameter_search_strategy == "random":
        if n_random_iterations <= 0:
            raise ValueError("n_random_iterations debe ser un número positivo")
        if n_random_iterations > 1000:
            logger.warning(f"n_random_iterations es muy alto ({n_random_iterations}). Considere usar un valor menor para mejorar el rendimiento.")

    # Inicializar variables de Bayesian search (se usarán si hyperparameter_search_strategy == "bayesian")
    n_trials = None
    n_initial_points = None
    timeout_seconds = None
    bayesian_search_params = {}
    optimization_time_seconds = None
    completed_trials = None
    best_score = None

    # Carga y preparación de datos
    df = load_and_validate_data(dataset_path, input_features, target_variable)
    X = df[input_features]
    y = df[target_variable]
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, split_ratios)

    # Configuración MLflow
    current_run = mlflow.active_run()
    if not current_run:
        raise RuntimeError("No hay un run activo de MLflow")

    run_id = current_run.info.run_id
    logger.info(f"Iniciando entrenamiento en run: {run_id}")

    # Registro de parámetros
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
        "problem_type": problem_type
    })

    # Entrenamiento del modelo
    if hyperparameter_search_strategy == "grid":
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "max_iter": [100, 200],
            "solver": ["lbfgs", "liblinear"]
        }
        grid_search = GridSearchCV(
            LogisticRegression(random_state=SEED),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=N_JOBS,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Guardar resultados completos de GridSearch
        cv_results_path = os.path.join(experiment_dir, "grid_search_results.csv")
        pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path)
        mlflow.log_artifact(cv_results_path, "grid_search")

    elif hyperparameter_search_strategy == "random":
        # Random search manual para Logistic Regression
        best_score = 0.0
        best_model = None
        best_params = None

        logger.info(f"Iniciando random search para Logistic Regression con {n_random_iterations} iteraciones...")

        for i in range(n_random_iterations):
            try:
                # Generar parámetros aleatorios
                random_params = generate_random_logistic_params(random_search_params)

                # Crear y entrenar modelo
                model_spec = LogisticRegression(**random_params)
                model_spec.fit(X_train, y_train)

                # Evaluar en validation set
                val_pred = model_spec.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_model = model_spec
                    best_params = random_params.copy()
                    best_params["val_accuracy"] = val_score

                if (i + 1) % 20 == 0:
                    logger.info(f"Progreso random search: {i+1}/{n_random_iterations}, Mejor accuracy: {best_score:.4f}")

            except Exception as e:
                logger.debug(f"Error con parámetros {random_params}: {e}")
                continue

        if best_model is None or best_params is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el random search")

        model = best_model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k != "val_accuracy"})
        mlflow.log_metric("best_val_accuracy", best_score)
        mlflow.log_metric("random_search_iterations", n_random_iterations)

    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend and convert format
        frontend_bayesian_params = data.get("bayesian_search_params", {})
        bayesian_search_params = convert_frontend_bayesian_params(frontend_bayesian_params)

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges
        default_ranges = {
            "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "max_iter": {"type": "int", "low": 100, "high": 1000},
            "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"]},
            "penalty": {"type": "categorical", "choices": ["l2", "none"]}
        }

        # Merge with user-provided ranges (user ranges override defaults)
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for Logistic Regression hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            C_config = param_ranges.get("C", default_ranges["C"])
            if C_config["type"] == "float":
                C = trial.suggest_float('C', C_config["low"], C_config["high"],
                                       log=C_config.get("log", True))

            max_iter_config = param_ranges.get("max_iter", default_ranges["max_iter"])
            max_iter = trial.suggest_int('max_iter', max_iter_config["low"],
                                        max_iter_config["high"])

            solver_config = param_ranges.get("solver", default_ranges["solver"])
            solver = trial.suggest_categorical('solver', solver_config["choices"])

            penalty_config = param_ranges.get("penalty", default_ranges["penalty"])
            penalty = trial.suggest_categorical('penalty', penalty_config["choices"])

            # Handle solver-penalty compatibility
            if solver == "liblinear" and penalty == "none":
                penalty = "l2"  # liblinear doesn't support penalty='none'
            if solver == "lbfgs" and penalty not in ["l2", "none"]:
                penalty = "l2"  # lbfgs only supports l2 and none

            try:
                # Train model with suggested parameters
                model_trial = LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver=solver,
                    penalty=penalty,
                    random_state=SEED,
                    n_jobs=N_JOBS
                )

                # Fit on training set
                model_trial.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"C={C:.4f}, solver={solver}, penalty={penalty}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')  # Penalty for failed trials

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,  # Fixed seed for reproducibility
            n_startup_trials=n_initial_points,  # Random exploration before Bayesian
            multivariate=False,  # Use independent TPE (simpler, more stable)
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',  # Minimize negative accuracy
            sampler=sampler,
            study_name=f"logistic_bayesian_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1  # Single-threaded for reproducibility
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value  # Convert back to positive accuracy

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        model = LogisticRegression(
            C=best_params_dict['C'],
            max_iter=best_params_dict['max_iter'],
            solver=best_params_dict['solver'],
            penalty=best_params_dict['penalty'],
            random_state=SEED,
            n_jobs=N_JOBS
        )
        model.fit(X_train, y_train)

        # Store best params for logging
        best_params = best_params_dict.copy()
        best_params["random_state"] = SEED

        # Log Bayesian search metadata to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items()}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })

    else:
        # Registrar parámetros manuales
        model_params = {
            "C": float(hyperparams.get("regularization", 1.0)),
            "max_iter": int(hyperparams.get("maxIter", 100)),
            "solver": hyperparams.get("solver", "lbfgs")
        }
        mlflow.log_params(model_params)
        
        model = LogisticRegression(
            random_state=SEED,
            **model_params
        )
        model.fit(X_train, y_train)
        best_params = model_params

    # Evaluación
    val_metrics, val_artifacts = evaluate_model(model, X_val, y_val, "val", problem_type, experiment_dir)
    test_metrics, test_artifacts = evaluate_model(model, X_test, y_test, "test", problem_type, experiment_dir)
    
    # Finalizar y registrar energía
    tracker.stop()
    energy_kwh, emissions_kg = log_energy_metrics(tracker)

    # Registro del modelo 
    signature = infer_signature(
        X_val,
        model.predict_proba(X_val) if problem_type == "multiclass" else model.predict(X_val)
    )
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model",
        signature=signature,
        registered_model_name=model_name,
        metadata={
            "dataset": os.path.basename(dataset_path),
            "features": input_features,
            "target": target_variable
        }
    )
    
    # Guardado local
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Actualizar metadatos
    client = MlflowClient()
    client.set_registered_model_tag(model_name, "task", "classification")
    client.update_registered_model(
        name=model_name,
        description=f"Modelo de regresión logística - {model_name}"
    )

    # Configuración del pipeline con parámetros completos
    pipeline_config = {
        "step": "train_logistic_regression",
        "model_name": model_name,
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameters": best_params,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "grid_search": {
            "use_grid_search": hyperparameter_search_strategy == "grid",
            "best_params": best_params if hyperparameter_search_strategy == "grid" else None,
            "param_grid": param_grid if hyperparameter_search_strategy == "grid" else None
        },
        "random_search": {
            "use_random_search": hyperparameter_search_strategy == "random",
            "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
            "random_search_params": random_search_params if hyperparameter_search_strategy == "random" else None,
            "best_params": best_params if hyperparameter_search_strategy == "random" else None,
        },
        "bayesian_search": {
            "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
            "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
            "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
            "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
            "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "artifacts": {
            "val": val_artifacts,
            "test": test_artifacts
        },
        "energy_metrics": {"energy_consumed_total_kWh": energy_kwh, "carbon_emission_kg": emissions_kg}
    }
    save_pipeline_config(experiment_dir, pipeline_config)

    return {
        "status": "Entrenamiento completado",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "run_id": run_id
    }

def train_mlp_model(dataset_path, data, experiment_dir):
    """
    Entrena un modelo MLP para clasificación binaria o multiclase.
    """
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=experiment_dir, save_to_file=False, allow_multiple_runs=True)
    tracker.start()

    # Extraer parámetros
    input_features = data["input_features"]
    target_variable = data["target_variable"]
    hyperparams = data.get("params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "MLP_Model")
    problem_type = data.get("problem_type", "binary").lower()

    # Soporte para nuevo formato de búsqueda de hiperparámetros con backward compatibility
    use_grid_search = data.get("use_grid_search", False)  # Deprecated, para backward compatibility
    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", None)

    # Si no se especifica hyperparameter_search_strategy, usar use_grid_search para determinar estrategia
    if hyperparameter_search_strategy is None:
        hyperparameter_search_strategy = "grid" if use_grid_search else "none"

    # Validar estrategia
    valid_strategies = ["none", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}. Recibido: {hyperparameter_search_strategy}")

    # Parámetros para random search
    n_random_iterations = data.get("n_random_iterations", 50)
    random_search_params = data.get("random_search_params", {})

    # Bayesian search variables (inicialización)
    n_trials = None
    n_initial_points = None
    timeout_seconds = None
    bayesian_search_params = {}
    optimization_time_seconds = None
    completed_trials = None
    best_score = None

    # Validar parámetros de random search
    if hyperparameter_search_strategy == "random":
        if n_random_iterations <= 0:
            raise ValueError("n_random_iterations debe ser un número positivo")
        if n_random_iterations > 1000:
            logger.warning(f"n_random_iterations es muy alto ({n_random_iterations}). Considere usar un valor menor para mejorar el rendimiento.")


    # Carga y preparación de datos
    df = load_and_validate_data(dataset_path, input_features, target_variable)
    X = df[input_features]
    y = df[target_variable]
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, split_ratios)

    # Configuración MLflow
    current_run = mlflow.active_run()
    if not current_run:
        raise RuntimeError("No hay un run activo de MLflow")

    run_id = current_run.info.run_id
    logger.info(f"Iniciando entrenamiento MLP en run: {run_id}")

    # Registro de parámetros
    mlflow.log_params({
        "model_type": "MLPClassifier",
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
        "problem_type": problem_type
    })

    # Entrenamiento del modelo
    if hyperparameter_search_strategy == "grid":
        param_grid = {
            "hidden_layer_sizes": [(4,), (10,), (10, 5), (100, 50, 10)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "max_iter": [200, 300]
        }
        grid_search = GridSearchCV(
            MLPClassifier(random_state=SEED),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=N_JOBS,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Guardar resultados
        cv_results_path = os.path.join(experiment_dir, "grid_search_results.csv")
        pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path)
        mlflow.log_artifact(cv_results_path, "grid_search")

    elif hyperparameter_search_strategy == "random":
        # Random search manual para MLP
        best_score = 0.0
        best_model = None
        best_params = None

        logger.info(f"Iniciando random search para MLP con {n_random_iterations} iteraciones...")

        for i in range(n_random_iterations):
            try:
                # Generar parámetros aleatorios
                random_params = generate_random_mlp_params(random_search_params)

                # Gestión de memoria: verbosidad según tamaño del dataset
                verbose_setting = 1 if X_train.shape[0] <= 10000 else 0
                random_params["verbose"] = verbose_setting

                # Crear y entrenar modelo
                model_spec = MLPClassifier(**random_params)
                model_spec.fit(X_train, y_train)

                # Evaluar en validation set
                val_pred = model_spec.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_model = model_spec
                    best_params = random_params.copy()
                    best_params["val_accuracy"] = val_score

                if (i + 1) % 20 == 0:
                    logger.info(f"Progreso random search: {i+1}/{n_random_iterations}, Mejor accuracy: {best_score:.4f}")

            except Exception as e:
                logger.debug(f"Error con parámetros {random_params}: {e}")
                continue

        if best_model is None or best_params is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el random search")

        model = best_model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k not in ["val_accuracy", "verbose"]})
        mlflow.log_metric("best_val_accuracy", best_score)
        mlflow.log_metric("random_search_iterations", n_random_iterations)

    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend and convert format
        frontend_bayesian_params = data.get("bayesian_search_params", {})
        bayesian_search_params = convert_frontend_bayesian_params(frontend_bayesian_params)

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana MLP (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges for MLP (using same defaults as random search)
        default_ranges = {
            "hidden_layer_sizes": {
                "type": "categorical",
                "choices": [(4,), (10,), (10, 5), (50,), (100,), (100, 50), (100, 50, 10)]
            },
            "activation": {"type": "categorical", "choices": ["relu", "tanh", "logistic"]},
            "learning_rate_init": {"type": "float", "low": 0.0001, "high": 0.1, "log": True},
            "solver": {"type": "categorical", "choices": ["adam", "sgd"]},
            "alpha": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
            "max_iter": {"type": "int", "low": 200, "high": 500}
        }

        # Merge with user-provided ranges
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for MLP hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            hidden_layer_sizes_config = param_ranges["hidden_layer_sizes"]
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes',
                hidden_layer_sizes_config["choices"])

            activation = trial.suggest_categorical('activation',
                param_ranges["activation"]["choices"])

            learning_rate_init = trial.suggest_float('learning_rate_init',
                param_ranges["learning_rate_init"]["low"],
                param_ranges["learning_rate_init"]["high"],
                log=param_ranges["learning_rate_init"].get("log", True))

            solver = trial.suggest_categorical('solver',
                param_ranges["solver"]["choices"])

            alpha = trial.suggest_float('alpha',
                param_ranges["alpha"]["low"],
                param_ranges["alpha"]["high"],
                log=param_ranges["alpha"].get("log", True))

            max_iter = trial.suggest_int('max_iter',
                param_ranges["max_iter"]["low"],
                param_ranges["max_iter"]["high"])

            try:
                # Memory-aware verbosity
                verbose_setting = 1 if X_train.shape[0] <= 10000 else 0

                # Train model
                model_trial = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=SEED,
                    verbose=verbose_setting
                )

                model_trial.fit(X_train, y_train)

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"hidden_layers={hidden_layer_sizes}, activation={activation}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,
            n_startup_trials=n_initial_points,
            multivariate=False,
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f"mlp_bayesian_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana MLP con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana MLP Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        verbose_setting = 1 if X_train.shape[0] <= 10000 else 0
        model = MLPClassifier(
            hidden_layer_sizes=best_params_dict['hidden_layer_sizes'],
            activation=best_params_dict['activation'],
            solver=best_params_dict['solver'],
            alpha=best_params_dict['alpha'],
            learning_rate_init=best_params_dict['learning_rate_init'],
            max_iter=best_params_dict['max_iter'],
            random_state=SEED,
            verbose=verbose_setting
        )
        model.fit(X_train, y_train)

        # Store best params for logging
        best_params = best_params_dict.copy()
        best_params["random_state"] = SEED
        best_params["verbose"] = verbose_setting

        # Log Bayesian search metadata to MLflow (exclude verbose from best_* params)
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items() if k != "verbose"}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })

    else:
        raw_hls = hyperparams.get("hidden_layer_sizes", "4")
        
        # Manejo robusto de hidden_layer_sizes
        if isinstance(raw_hls, (tuple, list)):
            hidden_layer_sizes = raw_hls
        elif isinstance(raw_hls, int):
            hidden_layer_sizes = (raw_hls,)
        elif isinstance(raw_hls, str):
            if "," in raw_hls:
                hidden_layer_sizes = tuple(map(int, raw_hls.split(",")))
            else:
                hidden_layer_sizes = (int(raw_hls),)
        else:
            raise ValueError(f"Formato no soportado para hidden_layer_sizes: {type(raw_hls)}")
        
        # Gestión de memoria: verbosidad según tamaño del dataset
        verbose_setting = 1 if X_train.shape[0] <= 10000 else 0
        
        # Registrar parámetros manuales
        model_params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": hyperparams.get("activation", "relu"),
            "solver": hyperparams.get("solver", "adam"),
            "max_iter": int(hyperparams.get("maxIter", 200))
        }
        mlflow.log_params(model_params)
        
        model = MLPClassifier(
            **model_params,
            random_state=SEED,
            shuffle=False,
            verbose=verbose_setting
        )
        model.fit(X_train, y_train)
        best_params = model_params

    # Evaluación
    val_metrics, val_artifacts = evaluate_model(model, X_val, y_val, "val", problem_type, experiment_dir)
    test_metrics, test_artifacts = evaluate_model(model, X_test, y_test, "test", problem_type, experiment_dir)
    
    # Finalizar y registrar energía
    tracker.stop()
    energy_kwh, emissions_kg = log_energy_metrics(tracker)

    # Registro del modelo con firma adecuada
    signature = infer_signature(
        X_val,
        model.predict_proba(X_val) if problem_type == "multiclass" else model.predict(X_val)
    )
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="mlp_model",
        signature=signature,
        registered_model_name=model_name,
        metadata={
            "dataset": os.path.basename(dataset_path),
            "features": input_features,
            "target": target_variable
        }
    )
    
    # Guardado local
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Actualizar metadatos
    client = MlflowClient()
    client.set_registered_model_tag(model_name, "task", "classification")
    client.update_registered_model(
        name=model_name,
        description=f"Modelo MLP para clasificación - {model_name}"
    )

    # Configuración del pipeline con parámetros completos
    pipeline_config = {
        "step": "train_mlp_model",
        "model_name": model_name,
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameters": best_params,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "grid_search": {
            "use_grid_search": hyperparameter_search_strategy == "grid",
            "best_params": best_params if hyperparameter_search_strategy == "grid" else None,
            "param_grid": param_grid if hyperparameter_search_strategy == "grid" else None
        },
        "random_search": {
            "use_random_search": hyperparameter_search_strategy == "random",
            "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
            "random_search_params": random_search_params if hyperparameter_search_strategy == "random" else None,
            "best_params": best_params if hyperparameter_search_strategy == "random" else None,
        },
        "bayesian_search": {
            "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
            "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
            "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
            "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
            "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
        },
        "problem_type": problem_type,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "artifacts": {
            "val": val_artifacts,
            "test": test_artifacts
        },
        "energy_metrics": {"energy_consumed_total_kWh": energy_kwh, "carbon_emission_kg": emissions_kg}
    }
    save_pipeline_config(experiment_dir, pipeline_config)

    return {
        "status": "Entrenamiento MLP completado",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "run_id": run_id
    }

def train_xgboost_model(dataset_path, data, experiment_dir):
    """
    Entrena un modelo XGBoost para clasificación binaria o multiclase.
    """
    # Crear directorio si no existe
    os.makedirs(experiment_dir, exist_ok=True)

    tracker = EmissionsTracker(output_dir=experiment_dir, save_to_file=False, allow_multiple_runs=True)
    tracker.start()

    # Extraer parámetros
    input_features = data["input_features"]
    target_variable = data["target_variable"]
    hyperparams = data.get("params", {})
    split_ratios = data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    model_name = data.get("model_name", "XGBoost_Model")
    problem_type = data.get("problem_type", "binary").lower()

    # Soporte para nuevo formato de búsqueda de hiperparámetros con backward compatibility
    use_grid_search = data.get("use_grid_search", False)  # Deprecated, para backward compatibility
    hyperparameter_search_strategy = data.get("hyperparameter_search_strategy", None)

    # Si no se especifica hyperparameter_search_strategy, usar use_grid_search para determinar estrategia
    if hyperparameter_search_strategy is None:
        hyperparameter_search_strategy = "grid" if use_grid_search else "none"

    # Validar estrategia
    valid_strategies = ["none", "grid", "random", "bayesian"]
    if hyperparameter_search_strategy not in valid_strategies:
        raise ValueError(f"hyperparameter_search_strategy debe ser uno de: {valid_strategies}. Recibido: {hyperparameter_search_strategy}")

    # Parámetros para random search
    n_random_iterations = data.get("n_random_iterations", 50)
    random_search_params = data.get("random_search_params", {})

    # Bayesian search variables (inicialización)
    n_trials = None
    n_initial_points = None
    timeout_seconds = None
    bayesian_search_params = {}
    optimization_time_seconds = None
    completed_trials = None
    best_score = None

    # Validar parámetros de random search
    if hyperparameter_search_strategy == "random":
        if n_random_iterations <= 0:
            raise ValueError("n_random_iterations debe ser un número positivo")
        if n_random_iterations > 1000:
            logger.warning(f"n_random_iterations es muy alto ({n_random_iterations}). Considere usar un valor menor para mejorar el rendimiento.")



    # Carga y preparación de datos
    df = load_and_validate_data(dataset_path, input_features, target_variable)
    X = df[input_features]
    y = df[target_variable].astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, split_ratios)

    # Configuración base
    base_params = {
        "objective": "binary:logistic" if problem_type == "binary" else "multi:softprob",
        "eval_metric": "logloss" if problem_type == "binary" else "mlogloss",
        "random_state": SEED,
        "tree_method": "hist",
        "use_label_encoder": False,
        "verbosity": 0
    }
    if problem_type == "multiclass":
        base_params["num_class"] = len(np.unique(y))

    # Configuración MLflow
    current_run = mlflow.active_run()
    if not current_run:
        raise RuntimeError("No hay un run activo de MLflow")

    run_id = current_run.info.run_id
    logger.info(f"Iniciando entrenamiento XGBoost en run: {run_id}")

    # Registro de parámetros
    mlflow.log_params({
        "model_type": "XGBClassifier",
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
        "problem_type": problem_type
    })

    # Entrenamiento del modelo
    if hyperparameter_search_strategy == "grid":
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
        grid_search = GridSearchCV(
            XGBClassifier(**base_params),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=N_JOBS,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Guardar resultados GridSearch
        cv_results_path = os.path.join(experiment_dir, "grid_search_results.csv")
        pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path)
        mlflow.log_artifact(cv_results_path, "grid_search")

    elif hyperparameter_search_strategy == "random":
        # Random search manual para XGBoost
        best_score = 0.0
        best_model = None
        best_params = None

        logger.info(f"Iniciando random search para XGBoost con {n_random_iterations} iteraciones...")

        for i in range(n_random_iterations):
            try:
                # Generar parámetros aleatorios
                random_params = generate_random_xgboost_params(random_search_params)

                # Combinar con parámetros base
                model_params = {**base_params, **random_params}

                # Crear y entrenar modelo
                model_spec = XGBClassifier(**model_params, early_stopping_rounds=10)
                model_spec.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )

                # Evaluar en validation set
                val_pred = model_spec.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_model = model_spec
                    best_params = random_params.copy()
                    best_params["val_accuracy"] = val_score
                    if hasattr(model_spec, 'best_iteration'):
                        best_params["best_iteration"] = model_spec.best_iteration

                if (i + 1) % 20 == 0:
                    logger.info(f"Progreso random search: {i+1}/{n_random_iterations}, Mejor accuracy: {best_score:.4f}")

            except Exception as e:
                logger.debug(f"Error con parámetros {random_params}: {e}")
                continue

        if best_model is None or best_params is None:
            raise RuntimeError("No se pudo entrenar ningún modelo en el random search")

        model = best_model
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if k != "val_accuracy"})
        mlflow.log_metric("best_val_accuracy", best_score)
        mlflow.log_metric("random_search_iterations", n_random_iterations)

    elif hyperparameter_search_strategy == "bayesian":
        # Extract Bayesian configuration
        bayesian_config = data.get("bayesian_config", {})
        n_trials = bayesian_config.get("n_trials", 50)
        n_initial_points = bayesian_config.get("n_initial_points", 10)
        timeout_seconds = bayesian_config.get("timeout_seconds", None)

        # Extract custom parameter ranges from frontend
        frontend_bayesian_params = data.get("bayesian_search_params", {})
        bayesian_search_params = convert_frontend_bayesian_params(frontend_bayesian_params)

        # Validate bayesian config
        if n_trials < 1:
            raise ValueError(f"n_trials must be at least 1, got {n_trials}")
        if n_initial_points >= n_trials:
            raise ValueError(
                f"n_initial_points ({n_initial_points}) must be less than n_trials ({n_trials})"
            )

        logger.info("="*60)
        logger.info("Configuración Búsqueda Bayesiana XGBoost (Optuna):")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  n_initial_points: {n_initial_points}")
        logger.info(f"  timeout_seconds: {timeout_seconds}")
        logger.info("="*60)

        # Define default parameter ranges for XGBoost
        default_ranges = {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0}
        }

        # Merge with user-provided ranges
        param_ranges = {**default_ranges, **bayesian_search_params}

        # Define Optuna objective function
        def objective(trial: Trial) -> float:
            """
            Optuna objective function for XGBoost hyperparameter optimization.

            Returns:
                float: Negative accuracy (for minimization)
            """
            # Suggest parameters based on configured ranges
            n_estimators = trial.suggest_int('n_estimators',
                param_ranges["n_estimators"]["low"],
                param_ranges["n_estimators"]["high"])

            max_depth = trial.suggest_int('max_depth',
                param_ranges["max_depth"]["low"],
                param_ranges["max_depth"]["high"])

            learning_rate = trial.suggest_float('learning_rate',
                param_ranges["learning_rate"]["low"],
                param_ranges["learning_rate"]["high"],
                log=param_ranges["learning_rate"].get("log", True))

            subsample = trial.suggest_float('subsample',
                param_ranges["subsample"]["low"],
                param_ranges["subsample"]["high"])

            colsample_bytree = trial.suggest_float('colsample_bytree',
                param_ranges["colsample_bytree"]["low"],
                param_ranges["colsample_bytree"]["high"])

            gamma = trial.suggest_float('gamma',
                param_ranges["gamma"]["low"],
                param_ranges["gamma"]["high"])

            min_child_weight = trial.suggest_int('min_child_weight',
                param_ranges["min_child_weight"]["low"],
                param_ranges["min_child_weight"]["high"])

            reg_alpha = trial.suggest_float('reg_alpha',
                param_ranges["reg_alpha"]["low"],
                param_ranges["reg_alpha"]["high"])

            reg_lambda = trial.suggest_float('reg_lambda',
                param_ranges["reg_lambda"]["low"],
                param_ranges["reg_lambda"]["high"])

            try:
                # Build model parameters (merge with base_params)
                model_params = {
                    **base_params,  # Includes objective, eval_metric, random_state, etc.
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "gamma": gamma,
                    "min_child_weight": min_child_weight,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda
                }

                # Train model with early stopping
                model_trial = XGBClassifier(**model_params, early_stopping_rounds=10)
                model_trial.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )

                # Evaluate on validation set
                val_pred = model_trial.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)

                logger.info(
                    f"Trial {trial.number}: accuracy={val_score:.4f}, "
                    f"n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate:.4f}"
                )

                # Return negative accuracy (Optuna minimizes)
                return -val_score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')

        # Create Optuna study with TPE sampler
        sampler = TPESampler(
            seed=SEED,
            n_startup_trials=n_initial_points,
            multivariate=False,
            consider_magic_clip=True,
            consider_endpoints=False
        )

        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=f"xgboost_bayesian_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        # Track optimization time
        optimization_start_time = time.time()

        # Run optimization
        logger.info(f"Iniciando Búsqueda Bayesiana XGBoost con Optuna TPESampler...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            n_jobs=1
        )

        optimization_time_seconds = time.time() - optimization_start_time

        # Validate results
        if study.best_trial is None or study.best_value == float('inf'):
            raise RuntimeError(
                "Búsqueda Bayesiana falló: Todos los trials retornaron errores. "
                "Verifique los rangos de parámetros y calidad de datos."
            )

        # Extract best parameters
        best_params_dict = study.best_params
        best_score = -study.best_value

        # Log optimization results
        logger.info("="*60)
        logger.info(f"Búsqueda Bayesiana XGBoost Completada")
        logger.info(f"  Mejor accuracy: {best_score:.4f}")
        logger.info(f"  Mejores parámetros: {best_params_dict}")
        completed_trials = len([t for t in study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"  Trials completados: {completed_trials}/{len(study.trials)}")
        logger.info(f"  Tiempo de optimización: {optimization_time_seconds:.2f} segundos")
        logger.info("="*60)

        # Train final model with best parameters
        model_params = {
            **base_params,
            **best_params_dict
        }
        model = XGBClassifier(**model_params, early_stopping_rounds=10)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0
        )

        # Store best params for logging (include best_iteration if available)
        best_params = best_params_dict.copy()
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            best_params["best_iteration"] = model.best_iteration

        # Log Bayesian search metadata to MLflow
        mlflow.log_params({
            "bayesian_n_trials": n_trials,
            "bayesian_n_initial_points": n_initial_points,
            "bayesian_optimization_metric": "accuracy",
            **{f"best_{k}": v for k, v in best_params_dict.items()}
        })

        mlflow.log_metrics({
            "bayesian_best_score": best_score,
            "bayesian_optimization_time_seconds": optimization_time_seconds,
            "bayesian_n_completed_trials": completed_trials
        })

    else:
        # Registrar parámetros manuales
        model_params = {
            "learning_rate": float(hyperparams.get("learning_rate", 0.1)),
            "n_estimators": int(hyperparams.get("n_estimators", 100)),
            "max_depth": int(hyperparams.get("max_depth", 3)),
            "subsample": float(hyperparams.get("subsample", 1.0)),
            "colsample_bytree": float(hyperparams.get("colsample_bytree", 1.0))
        }
        mlflow.log_params(model_params)

        model = XGBClassifier(
            **base_params,
            **model_params,
            early_stopping_rounds=10
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10  # Loggear progreso
        )
        best_params = model_params
        
        # Registrar mejor iteración
        if hasattr(model, 'best_iteration'):
            mlflow.log_metric("best_iteration", model.best_iteration)
            best_params["best_iteration"] = model.best_iteration

    # Evaluación
    val_metrics, val_artifacts = evaluate_model(model, X_val, y_val, "val", problem_type, experiment_dir)
    test_metrics, test_artifacts = evaluate_model(model, X_test, y_test, "test", problem_type, experiment_dir)
    
    # Finalizar y registrar energía
    tracker.stop()
    energy_kwh, emissions_kg = log_energy_metrics(tracker)

    # Registro del modelo con firma adecuada
    signature = infer_signature(
        X_val,
        model.predict_proba(X_val) if problem_type == "multiclass" else model.predict(X_val)
    )
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="xgboost_model",
        signature=signature,
        registered_model_name=model_name,
        metadata={
            "dataset": os.path.basename(dataset_path),
            "features": input_features,
            "target": target_variable
        }
    )
    
    # Guardado local
    model_path = os.path.join(experiment_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Actualizar metadatos
    client = MlflowClient()
    client.update_registered_model(
        name=model_name,
        description=f"Modelo XGBoost para clasificación - {model_name}"
    )

    # Configuración del pipeline con parámetros completos
    pipeline_config = {
        "step": "train_xgboost",
        "model_name": model_name,
        "input_features": input_features,
        "target_variable": target_variable,
        "split_ratios": split_ratios,
        "hyperparameters": best_params,
        "hyperparameter_search_strategy": hyperparameter_search_strategy,
        "grid_search": {
            "use_grid_search": hyperparameter_search_strategy == "grid",
            "best_params": best_params if hyperparameter_search_strategy == "grid" else None,
            "param_grid": param_grid if hyperparameter_search_strategy == "grid" else None
        },
        "random_search": {
            "use_random_search": hyperparameter_search_strategy == "random",
            "n_random_iterations": n_random_iterations if hyperparameter_search_strategy == "random" else None,
            "random_search_params": random_search_params if hyperparameter_search_strategy == "random" else None,
            "best_params": best_params if hyperparameter_search_strategy == "random" else None,
        },
        "bayesian_search": {
            "use_bayesian_search": hyperparameter_search_strategy == "bayesian",
            "n_trials": n_trials if hyperparameter_search_strategy == "bayesian" else None,
            "n_initial_points": n_initial_points if hyperparameter_search_strategy == "bayesian" else None,
            "timeout_seconds": timeout_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "bayesian_search_params": bayesian_search_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_params": best_params if hyperparameter_search_strategy == "bayesian" else None,
            "best_score": best_score if hyperparameter_search_strategy == "bayesian" else None,
            "optimization_time_seconds": optimization_time_seconds if hyperparameter_search_strategy == "bayesian" else None,
            "n_completed_trials": completed_trials if hyperparameter_search_strategy == "bayesian" else None
        },
        "problem_type": problem_type,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "artifacts": {
            "val": val_artifacts,
            "test": test_artifacts
        },
        "energy_metrics": {"energy_consumed_total_kWh": energy_kwh, "carbon_emission_kg": emissions_kg}
    }
    save_pipeline_config(experiment_dir, pipeline_config)

    return {
        "status": "Entrenamiento XGBoost completado",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": model_path,
        "run_id": run_id
    }