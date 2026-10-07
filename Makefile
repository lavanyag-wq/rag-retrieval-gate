.PHONY: lint test experiments receipts verify report

lint:
	ruff check src tests tools experiments

test:
	pytest --junitxml=reports/junit.xml --cov=rag_retrieval_gate --cov-report=json:reports/coverage.json --cov-report=term

experiments:
	python experiments/exp_regression.py
	python experiments/exp_resolution.py
	python experiments/exp_noise_floor.py

receipts:
	python tools/collect_metrics.py --skip-tests
	python tools/check_numbers.py

verify: lint test experiments receipts
