.PHONY: setup test run design clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	. .venv/bin/activate || true; pytest -q

run:
	. .venv/bin/activate || true; python "scripts/Main_code.py"

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ project_output reports models source_data data/processed .venv
