#!/bin/bash

set -e

echo "Running Black..."
black src/hawkears

echo "Running Ruff..."
ruff check src/hawkears

echo "Running Mypy..."
mypy src/hawkears
