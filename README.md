# Adamic LLM

This project was made as part of the study: 
The Cost-Benefit off Translation
Improving Large Language Models Multilingual
Capability with Adamic Prompting

## UV

This project uses uv. It's a modern dependency management
tool.

To run the project use this set of commands:

```bash
uv sync --locked
uv run -m adamic_llm
```

This will start the server on the configured host.

You can find swagger documentation at `/api/docs`.

You can read more about uv here: https://docs.astral.sh/ruff/

## Docker

You can start the project with docker using this command:

```bash
docker-compose up --build
```

If you want to develop in docker with autoreload and exposed ports add `-f deploy/docker-compose.dev.yml` to your docker command.
Like this:

```bash
docker-compose -f docker-compose.yml -f deploy/docker-compose.dev.yml --project-directory . up --build
```

This command exposes the web application on port 8000, mounts current directory and enables autoreload.

But you have to rebuild image every time you modify `uv.lock` or `pyproject.toml` with this command:

```bash
docker-compose build
```

## Project structure

```bash
$ tree "adamic_llm"
adamic_llm
├── db
│   ├── dao
│   │   ├── dummy_dao.py
│   │   └── __pycache__
│   │       └── dummy_dao.cpython-313.pyc
│   ├── dependencies.py
│   ├── models
│   │   ├── dummy_model.py
│   │   └── __pycache__
│   │       └── dummy_model.cpython-313.pyc
│   └── __pycache__
│       └── dependencies.cpython-313.pyc
├── gunicorn_runner.py
├── __init__.py
├── log.py
├── __main__.py
├── __pycache__
│   ├── gunicorn_runner.cpython-313.pyc
│   ├── __init__.cpython-313.pyc
│   ├── log.cpython-313.pyc
│   ├── __main__.cpython-313.pyc
│   └── settings.cpython-313.pyc
├── services
│   ├── adamic_prompting
│   │   ├── adamic_history.py
│   │   ├── configuration.py
│   │   ├── graph.py
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── __pycache__
│   │   │   ├── adamic_history.cpython-313.pyc
│   │   │   ├── configuration.cpython-313.pyc
│   │   │   ├── graph.cpython-313.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   ├── prompts.cpython-313.pyc
│   │   │   ├── state.cpython-313.pyc
│   │   │   └── utils.cpython-313.pyc
│   │   ├── state.py
│   │   └── utils.py
│   ├── chat
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   └── service.cpython-313.pyc
│   │   └── service.py
│   ├── google_translate
│   │   ├── dependency.py
│   │   ├── __init__.py
│   │   ├── lifespan.py
│   │   └── __pycache__
│   │       ├── dependency.cpython-313.pyc
│   │       ├── __init__.cpython-313.pyc
│   │       └── lifespan.cpython-313.pyc
│   ├── graph
│   │   ├── graph_registry.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── graph_registry.cpython-313.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   └── runner.cpython-313.pyc
│   │   └── runner.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   └── service.cpython-313.pyc
│   │   └── service.py
│   ├── __pycache__
│   │   └── __init__.cpython-313.pyc
│   └── redis
│       ├── dependency.py
│       ├── __init__.py
│       ├── lifespan.py
│       └── __pycache__
│           ├── dependency.cpython-313.pyc
│           ├── __init__.cpython-313.pyc
│           └── lifespan.cpython-313.pyc
├── settings.py
├── static
│   └── docs
│       ├── redoc.standalone.js
│       ├── swagger-ui-bundle.js
│       └── swagger-ui.css
├── utils
│   ├── __init__.py
│   ├── message.py
│   └── __pycache__
│       ├── __init__.cpython-313.pyc
│       └── message.cpython-313.pyc
└── web
    ├── api
    │   ├── chat
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   ├── schema.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   ├── schema.py
    │   │   └── views.py
    │   ├── docs
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   └── views.py
    │   ├── dummy
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   ├── schema.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   ├── schema.py
    │   │   └── views.py
    │   ├── echo
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   ├── schema.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   ├── schema.py
    │   │   └── views.py
    │   ├── __init__.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   ├── schemas.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   ├── schemas.py
    │   │   └── views.py
    │   ├── monitoring
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   └── views.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-313.pyc
    │   │   └── router.cpython-313.pyc
    │   ├── redis
    │   │   ├── __init__.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-313.pyc
    │   │   │   ├── schema.cpython-313.pyc
    │   │   │   └── views.cpython-313.pyc
    │   │   ├── schema.py
    │   │   └── views.py
    │   └── router.py
    ├── application.py
    ├── __init__.py
    ├── lifespan.py
    ├── openai_server.py
    └── __pycache__
        ├── application.cpython-313.pyc
        ├── __init__.cpython-313.pyc
        ├── lifespan.cpython-313.pyc
        └── openai_server.cpython-313.pyc
```

## Configuration

This application can be configured with environment variables.

You can create `.env` file in the root directory and place all
environment variables here. 

All environment variables should start with "ADAMIC_LLM_" prefix.

For example if you see in your "adamic_llm/settings.py" a variable named like
`random_parameter`, you should provide the "ADAMIC_LLM_RANDOM_PARAMETER" 
variable to configure the value. This behaviour can be changed by overriding `env_prefix` property
in `adamic_llm.settings.Settings.Config`.

An example of .env file:
```bash
ADAMIC_LLM_RELOAD="True"
ADAMIC_LLM_PORT="8000"
ADAMIC_LLM_ENVIRONMENT="dev"
```

You can read more about BaseSettings class here: https://pydantic-docs.helpmanual.io/usage/settings/

## Pre-commit

To install pre-commit simply run inside the shell:
```bash
pre-commit install
```

pre-commit is very useful to check your code before publishing it.
It's configured using .pre-commit-config.yaml file.

By default it runs:
* mypy (validates types);
* ruff (spots possible bugs);


You can read more about pre-commit here: https://pre-commit.com/


## Running tests

If you want to run it in docker, simply run:

```bash
docker-compose run --build --rm api pytest -vv . && docker-compose down
```

For running tests on your local machine.
1. you need to start a database.

I prefer doing it with docker:
```
docker run -p "5432:5432" -e "POSTGRES_PASSWORD=adamic_llm" -e "POSTGRES_USER=adamic_llm" -e "POSTGRES_DB=adamic_llm" postgres:18.1-bookworm
```


2. Run the pytest.
```bash
pytest -vv .
```
