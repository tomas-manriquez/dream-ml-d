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

from django.urls import path
from . import views

urlpatterns = [
    path('start-mlflow/', views.start_mlflow, name='start_mlflow'),
    path('init-dvc/', views.init_dvc, name='init_dvc'),
    path('configure-dvc-remote/', views.configure_dvc_remote, name='configure_dvc_remote'),
    path('upload-and-clean-csv/', views.upload_and_clean_csv, name='upload_and_clean_csv'),
    path('get-pipeline-config/', views.get_pipeline_config, name='get_pipeline_config'),
    path('run-pipeline/', views.run_pipeline, name='run_pipeline'),
    path('generate-eda/', views.generar_reporte_eda, name='generate_eda'),
    path('start-jupyter/', views.start_jupyter, name='start_jupyter'),
    path('analyze-csv/', views.analyze_csv, name='analyze_csv'),
    path('encode-csv/', views.encode_csv, name='encode_csv'),
    path('create-experiment/', views.create_experiment, name='create_experiment'),  
    path('train-model/', views.train_model, name='train_model'),
    path('get-experiment-summary/', views.get_experiment_summary, name='get_experiment_summary'),  
]
