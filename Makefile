.PHONY: setup lab app clean

setup:
	uv sync

lab:
	uv run jupyter lab

app:
	uv run streamlit run app.py

clean:
	rm -rf .venv
