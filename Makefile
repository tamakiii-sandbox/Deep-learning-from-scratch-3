.PHONY: help test check mypy format format-check

export PYTHONPATH := .

help:
	@cat $(firstword $(MAKEFILE_LIST))

test:
	pytest tests

check: \
	mypy \
	format-check

mypy:
	mypy --explicit-package-bases steps

format:
	black .

format-check:
	black --check .
