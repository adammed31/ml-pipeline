.PHONY: install api app mlflow train help

PYTHON = python
DATA ?= data/raw/dataset.csv
TARGET ?= target

help:
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make api              Start FastAPI (port 8000)"
	@echo "  make app              Start Streamlit dashboard (port 8501)"
	@echo "  make mlflow           Start MLflow UI (port 5001)"
	@echo "  make train DATA=path/to/data.csv TARGET=col_name  Train via CLI"
	@echo "  make notebook         Start Jupyter Lab"

install:
	pip install -r requirements.txt

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py --server.port 8501

mlflow:
	mlflow ui --backend-store-uri mlruns --port 5001

train:
	$(PYTHON) train.py --data $(DATA) --target $(TARGET)

notebook:
	jupyter lab --notebook-dir=notebooks
