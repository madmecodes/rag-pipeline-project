.PHONY: install data ingest golden eval test serve up

install:
	uv sync

data:
	kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets -p data/raw --unzip

ingest:
	uv run python -m rag.ingest

golden:
	uv run python -m eval.build_golden --n 150

eval:
	uv run python -m eval.run_eval --gen-n 30

test:
	uv run pytest -q

serve:
	uv run uvicorn rag.api:app --port 8000

up:
	docker compose up --build -d && docker compose exec api uv run --no-sync python -m rag.ingest
