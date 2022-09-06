#!/bin/bash

case "$1" in
	li|lizard)
		PYTHONPATH=./src python3 ./tests/test_lizard.py
		;;
	c|cluster)
		det -m "https://dt1.f4.htw-berlin.de:8443" -u bschilling experiment create "$2" ./src
		;;
	lc|localcluster)
		det -m "http://localhost:8080" -u admin experiment create ./configs/test_tiny.yaml ./src
		;;
	t|tensorboard)
		if [ -z "$2" ]; then
			echo "missing experiment id"
			exit 1
		fi
		det tensorboard start --config "resources.agent_label=dt-cluster" "$2"
		;;
	b)
		PYTHONPATH=./src ipython ./tests/test_banana.py
		;;
	d|default)
		shift
		if [ "$1" == "--dry" ]; then
			dry='--dry'
			shift
		fi

		case "$1" in
			d|default)
				source ./p/default_patches.sh
				;;
			e|early)
				source ./p/early_patches.sh
				;;
			s|scaleup)
				source ./p/scaleup_patches.sh
				;;
			c|center)
				source ./p/center_patches.sh
				;;
			*)
				echo "ERROR: unknown patch configuration: $1"
				exit 1
				;;
		esac

		python3 ./utils/start_experiment.py -v $dry "$base_config" "${patches[@]}"
		;;
	o|orig)
		PYTHONPATH=./src python3 ./tests/test_orig.py
		;;
	m|model)
		PYTHONPATH=./src python3 ./tests/test_model.py
		;;
	r)
		shift
		if [ "$1" == "--dry" ]; then
			dry='--dry'
			shift
		fi
		python3 ./utils/start_experiment.py -v $dry "./configs/base_config.yaml" "$@"
		;;
	f|function)
		PYTHONPATH=./src python3 ./tests/test_function.py
		;;
	g|gpu)
		PYTHONPATH=./src python3 ./tests/test_gpu.py
		;;
	pr)
		PYTHONPATH=./src python3 ./tests/test_prcurve.py
		;;
	map)
		PYTHONPATH=./src python3 ./tests/test_map.py
		;;
	*)
		echo "invalid run option"
		;;
esac
