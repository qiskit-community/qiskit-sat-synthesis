.PHONY: lint format test

SOURCES = qiskit_sat_synthesis/ tests/ examples/ experiments/ setup.py

lint:
	black --check $(SOURCES)
	ruff check $(SOURCES)

format:
	black $(SOURCES)
	ruff check --fix $(SOURCES)

test:
	pytest tests/
