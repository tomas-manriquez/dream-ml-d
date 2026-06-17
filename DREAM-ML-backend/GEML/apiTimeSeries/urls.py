from django.urls import path

from . import views
from api import views as CLViews

urlpatterns = [
    # ex. /api/ts/
    path("", views.index, name="index"),
    # ex. api/ts/analyze-csv/, formData
    path("analyze-csv/", CLViews.analyze_csv, name="analyze-csv"),
    # ex. api/ts/preview-date-standardization/, formData
    path("preview-date-standardization/", views.preview_date_standardization, name="preview-date-standardization"),
    # ex. api/ts/upload-and-clean-csv/, formData
    path("upload-and-clean-csv/", views.upload_and_clean_csv, name="upload-and-clean-csv"),
    # ex. api/ts/generate-eda/, formData
    path("generate-eda/", views.generate_eda_report, name="generate-eda"),
    # ex. api/ts/encode-csv/, formData
    path("encode-csv/", views.encode_csv, name="encode-csv"),
    # ex. api/ts/train-model/, formData
    path("train-model/", views.train_model, name="train-model"),
    # ex. api/ts/run-pipeline/, formData
    path("run-pipeline", views.run_pipeline, name="run-pipeline"),
]