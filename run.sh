#!/bin/bash

case "$1" in
	t)
		PYTHONPATH=./src python3 ./tests/test_gpu.py
		;;
	li|lizard)
		PYTHONPATH=./src python3 ./tests/test_lizard.py
		;;
	l|local)
		PYTHONPATH=./src python3 ./src/local_test.py
		;;
	*)
		PYTHONPATH=./src python3 ./tests/test_function.py
		;;
esac
