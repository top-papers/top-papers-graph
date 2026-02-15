.PHONY: up down lint test ui

.PHONY: docker-build docker-up docker-down docker-shell docker-run

up:
	docker compose up -d

down:
	docker compose down -v

docker-build:
	docker compose build app

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

docker-shell:
	docker compose exec app bash

# Example:
#   make docker-run ARGS='--query "graph neural network survey" --sources all --top-papers 20'
docker-run:
	docker compose exec app top-papers-graph run $(ARGS)

lint:
	ruff check .

test:
	pytest -q

ui:
	streamlit run -m scireason.ui.streamlit_app


paper:
	cd paper/agents4science && latexmk -pdf -interaction=nonstopmode main.tex


refresh-feedback:
	python -m scireason.cli refresh-feedback
