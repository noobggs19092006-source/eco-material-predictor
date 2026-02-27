VENV    := venv/bin/python
PYTEST  := venv/bin/pytest
PROJECT := eco-material-predictor

.PHONY: install train evaluate predict test clean help

## install: create venv and install all dependencies
install:
	@bash setup.sh

# Guard: fail fast with a helpful message if venv or packages are missing
_check_venv:
	@if [ ! -f "$(VENV)" ]; then \
		echo ""; \
		echo "❌  venv not found. Run:  bash setup.sh"; \
		echo ""; \
		exit 1; \
	fi
	@if [ ! -f "$(PYTEST)" ]; then \
		echo ""; \
		echo "❌  Dependencies not installed. Run:  bash setup.sh"; \
		echo ""; \
		exit 1; \
	fi

## train: run the training pipeline (data prep + model fitting)
train: _check_venv
	$(VENV) src/train.py

## evaluate: evaluate trained model and generate result plots
evaluate: _check_venv
	$(VENV) src/evaluate.py

## predict: launch the interactive CLI predictor
predict: _check_venv
	$(VENV) src/cli.py

## app: launch FastAPI backend + React frontend
app: _check_venv
	@echo "🚀 Starting API on :8000 and frontend on :5173 ..."
	@$(VENV) -m uvicorn src.api:app --reload --port 8000 & \
	cd frontend && npm run dev


## test: run pytest unit tests
test: _check_venv
	$(PYTEST) tests/ -v --tb=short

## clean: remove generated model, results, and processed data
clean:
	rm -rf models/*.pkl \
	       results/prediction_*.csv \
	       results/molecule_*.pdb \
	       results/*.png \
	       results/*.txt \
	       data/processed/*.csv
	@echo "🧹 Cleaned generated artefacts."

## help: list all available make targets
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
