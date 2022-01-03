#!/usr/bin/env bash
# runUnitTests.sh
# export COVERAGE_DEBUG=process,config
black src
python -m pytest --doctest-modules --mypy --pylint -vv src/
