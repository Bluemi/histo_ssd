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
	r|cluster)
		det -m "https://dt1.f4.htw-berlin.de:8443" -u bschilling experiment create ./configs/test_tiny.yaml ./src
		;;
	b)
		PYTHONPATH=./src ipython ./tests/test_banana.py
		;;
	d|data)
		PYTHONPATH=./src python3 ./tests/test_datasets.py
		;;
	*)
		PYTHONPATH=./src python3 ./tests/test_function.py
		;;
esac
