#!/bin/bash

case "$1" in
	t)
		PYTHONPATH=./src python3 ./tests/test_lizard.py
		;;
	*)
		echo "invalid input"
		;;
esac
