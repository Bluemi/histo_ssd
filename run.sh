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
		patches=(
			# p/cluster/pepper.yaml
			p/cluster/pepper_few.yaml
			# p/cluster/dt.yaml

			p/dataset/lizard.yaml
			p/dataset/ignore_rare_classes.yaml
			# p/dataset/scale_up.yaml

			p/model/vgg16.yaml
			# p/model/vgg16early.yaml

			# p/augmentation/test_rotate_flip.yaml
			# p/hparams/lr_narrow.yaml
			# p/hparams/lr.yaml
			# p/hparams/sizes.yaml
			# p/hparams/sizes_small.yaml
			# p/hparams/negative_ratio_narrow.yaml
			# p/hparams/image_stride.yaml
			# p/hparams/smoothl1.yaml
			# p/hparams/nms_threshold.yaml
			# p/hparams/bbox_loss_scale.yaml

			# p/metrics/loss.yaml
			p/metrics/map.yaml
			# p/metrics/write_predictions.yaml

			# p/checkpointing/save_checkpoints.yaml
			p/checkpointing/load_checkpoint.yaml
			# p/checkpointing/load_checkpoint_scaleup.yaml

			# p/searcher/adaptive_asha.yaml
			# p/searcher/adaptive_asha_few.yaml
			# p/searcher/random.yaml

			p/profiling/clock.yaml
		)

		base_config="./configs/base_config.yaml"
		# base_config="./configs/scale_up_config.yaml"

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
	*)
		echo "invalid run option"
		;;
esac
