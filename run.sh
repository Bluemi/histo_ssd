#!/bin/bash

case "$1" in
	t)
		PYTHONPATH=./src python3 ./tests/test_gpu.py
		;;
	l)
		PYTHONPATH=./src python3 ./tests/test_lizard.py
		;;
	*)
		PYTHONPATH=./src python3 ./tests/test_function.py
		;;
esac
