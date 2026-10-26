default:
    @just --list

setup:
    git submodule update --init --recursive
    uv sync --python 3.12

verify-usd:
    uv run python scripts/generate_newton_usd.py --verify-only

generate-usd *args:
    uv run python scripts/generate_newton_usd.py {{args}}

generate-anymal-usd *args:
    uv run python scripts/generate_anymal_newton_usd.py {{args}}

generate-honeycomb-collisions:
    uv run python scripts/generate_honeycomb_collisions.py
