.PHONY: test lint typecheck check install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

typecheck:
	mypy src/openrouter_agent/

check: lint typecheck test
