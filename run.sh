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
		python3 ./utils/start_experiment.py -v $dry "./configs/base_config.yaml" p/augmentation/test_rotate_flip.yaml p/cluster/pepper.yaml p/dataset/lizard.yaml p/hparams/lr_narrow.yaml p/hparams/negative_ratio_narrow.yaml p/hparams/image_stride.yaml p/hparams/smoothl1.yaml p/hparams/nms_threshold.yaml p/metrics/loss.yaml p/model/vgg16.yaml p/searcher/random.yaml
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
	*)
		echo "invalid run option"
		;;
esac
