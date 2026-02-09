.PHONY: up down lint test ui

up:
	docker compose up -d

down:
	docker compose down -v

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
