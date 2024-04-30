.PHONY: help install shell

help:
	@cat $(firstword $(MAKEFILE_LIST))

install:
	poetry install --no-root

shell:
	poetry shell
