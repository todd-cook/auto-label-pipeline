#!/usr/bin/env bash
# runUnitTests.sh
# export COVERAGE_DEBUG=process,config
black src
python -m pytest --doctest-modules --cov-report=term --mypy --cov-config=.coveragerc \
--pylint -vv src/  --cov=src
find . -name '.coverage' -type f -delete
find . -name '.coverage.*' -type f -delete
