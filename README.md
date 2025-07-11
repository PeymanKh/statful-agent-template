# Langgraph agent with tool

This is production implementation of an LLM with tools.
This repo can be used as a reference for more complex LLM-based applications.

## Structure
- `src/chains/`: handling chains 
- `src/config/`: project settings and .env loading
- `src/graphs/`: graph implementation
- `src/models/`: ChatModel implementation
- `src/nodes/`: graph's node definition
- `src/prompts/`: prompts variables
- `src/schemas/`: graph state and database models
- `src/tools/`: llm tools definition
- `src/utils/`: utilities and helper functions
- `tests/`: Unit and integration tests
- `main.py`: Entry point

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```