.PHONY: lint

lint:
	black .
	ruff check . --fix
	mypy .
