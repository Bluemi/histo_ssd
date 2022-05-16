#!/bin/bash

case "$1" in
	t)
		python3 ./tests/test1.py
		;;
	*)
		echo "invalid input"
		;;
esac
