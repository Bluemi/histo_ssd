base_config="./configs/center_config.yaml"
patches=(
	# p/cluster/pepper.yaml
	# p/cluster/pepper_few.yaml
	p/cluster/dt.yaml

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
	# p/hparams/sizes_med.yaml
	# p/hparams/negative_ratio_narrow.yaml
	# p/hparams/image_stride.yaml
	# p/hparams/smoothl1.yaml
	# p/hparams/nms_threshold.yaml
	# p/hparams/bbox_loss_scale.yaml
	# p/hparams/iou_match_threshold.yaml

	p/metrics/loss.yaml
	# p/metrics/map.yaml
	p/metrics/write_predictions.yaml
	# p/metrics/write_few_predictions.yaml

	p/checkpointing/save_checkpoints.yaml
	# p/checkpointing/load_checkpoint.yaml
	# p/checkpointing/load_checkpoint_scaleup.yaml
	# p/checkpointing/load_checkpoint_early.yaml

	# p/searcher/adaptive_asha.yaml
	# p/searcher/adaptive_asha_few.yaml
	# p/searcher/random.yaml

	p/profiling/clock.yaml
)
