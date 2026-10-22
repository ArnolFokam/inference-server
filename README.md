# inference-server

An inference server for tiny LLMs, built from scratch: model loading, tokenization, KV-cached generation, and an HTTP API, with minimal dependencies.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e '.[dev]'

## Run the tests

    pytest

## Layout

    src/inference/    the package
    tests/            the tests
