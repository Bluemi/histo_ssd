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
	c|cluster)
		det -m "https://dt1.f4.htw-berlin.de:8443" -u bschilling experiment create --follow-first-trial ./configs/test_tiny.yaml ./src
		;;
	lc|localcluster)
		det -m "http://localhost:8080" -u admin experiment create --follow-first-trial ./configs/test_tiny.yaml ./src
		;;
	d|det)
		det -m "https://dt1.f4.htw-berlin.de:8443" -u bschilling "$@"
		;;
	b)
		PYTHONPATH=./src ipython ./tests/test_banana.py
		;;
	d|data)
		PYTHONPATH=./src python3 ./tests/test_datasets.py
		;;
	o|orig)
		PYTHONPATH=./src python3 ./tests/test_orig.py
		;;
	*)
		PYTHONPATH=./src python3 ./tests/test_function.py
		;;
esac
