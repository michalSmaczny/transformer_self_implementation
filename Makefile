.PHONY: lint

lint:
	ruff check . --fix
	black .
	mypy .
