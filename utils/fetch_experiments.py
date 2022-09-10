import concurrent.futures
import pathlib
import argparse
import sys
from typing import Tuple, List, Dict, Any, Optional

from determined.common.api.errors import APIException
from determined.experimental import Determined, TrialReference, ExperimentReference
from determined.common import api
import pandas as pd
from ruamel.yaml import YAML

PARENT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_PATH = PARENT_DIR / 'data.csv'
CREDENTIALS_PATH = PARENT_DIR / 'credentials.yaml'
DET_MASTER = 'https://dt1.f4.htw-berlin.de:8443'
yaml = YAML()
yaml.indent(sequence=4)


HPARAM_KEYS = {
    # 'optimizer': 'optimizer',
    'learning_rate': 'learning_rate',
    'l2_regularization': 'l2_regularization',
    'momentum': 'momentum',
    # 'image_prediction_score_threshold': 'image_prediction_score_threshold',
    # 'image_prediction_max_images': 'image_prediction_max_images',
    'negative_ratio': 'negative_ratio',
    'min_anchor_size': 'min_anchor_size',
    'max_anchor_size': 'max_anchor_size',
    'nms_threshold': 'nms_threshold',
    'aug_norm': 'aug_norm',
    'aug_rotate': 'aug_rotate',
    'aug_flip': 'aug_flip',
    'image_stride': 'image_stride',
    # 'enable_class_metrics': 'enable_class_metrics',
    # 'use_clock': 'use_clock',
    'use_smooth_l1': 'use_smooth_l1',
    # 'max_eval_time': 'max_eval_time',
    'bbox_loss_scale': 'bbox_loss_scale',
    'dataset_split_size': 'dataset_split_size',
    'ignore_classes': 'ignore_classes',
    'hnm_norm_per_batch': 'hnm_norm_per_batch',
    'dataset_image_size': 'dataset_image_size',
    'dataset': 'dataset',
    'warmup_batches': 'warmup_batches',
    'backbone_arch': 'backbone_arch',
    'iou_match_threshold': 'iou_match_threshold',
    'model_image_size': 'model_image_size',
    'pretrained': 'pretrained',
    'shuffle_validation': 'shuffle_validation',
    'force_one_class': 'force_one_class',
}

METRIC_KEYS = {
    'f1_score': 'f1_score',
    'recall': 'recall',
    'precision': 'precision',
    'map': 'map',
    'loss': 'loss',
    'bbox_loss': 'bbox_loss',
    'cls_loss': 'cls_loss',
    'map_50': 'map_50',
    'map_75': 'map_75',
}


def get_credentials():
    if not CREDENTIALS_PATH.is_file():
        raise ValueError(
            'No credentials found. Template created at \"{}\". Please fill in credentials.'.format(CREDENTIALS_PATH)
        )

    with open(CREDENTIALS_PATH, 'r') as credentials_file:
        credentials = yaml.load(credentials_file)
    if credentials['username'] == '<username>':
        raise ValueError(
            'Credentials are invalid. Please fill in correct credentials at \"{}\".'.format(CREDENTIALS_PATH)
        )
    return credentials


def get_dict_recursive(d, keys):
    for key in keys:
        if isinstance(d, dict):
            if key in d:
                d = d[key]
        else:
            return None
    return d


def get_determined_client() -> Determined:
    """
    Reads credentials and creates determined client.
    """
    credentials = get_credentials()
    return Determined(master=DET_MASTER, user=credentials['username'], password=credentials['password'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_id', type=int, nargs='+', help='The experiment ids to fetch')
    parser.add_argument(
        '--merge', action='store_true',
        help='Merge new experiment info with existing csv data. Otherwise old data is discarded'
    )
    return parser.parse_args()


def insert_dict(source: dict, target: dict, keys: dict) -> Tuple[List[str], List[str]]:
    """
    Copies all keys from source to target dictionary.
    Returns list of keys which are missing in source and a list with keys in source but not in keys.
    """
    missing_keys = []
    for source_key, target_key in keys.items():
        if source_key in source:
            target[target_key] = source[source_key]
        elif target_key not in target:
            target[target_key] = None
            missing_keys.append(source_key)
    ignored_keys = list(filter(lambda k: k not in keys, source.keys()))
    return missing_keys, ignored_keys


class FetchTrialRawArgs:
    def __init__(self, user: str, trial_id: int):
        self.master = DET_MASTER
        self.user = user
        self.trial_id = trial_id


@api.authentication.required
def _fetch_trial_raw(args: FetchTrialRawArgs):
    r = api.get(args.master, "trials/{}".format(args.trial_id))
    return r.json()


def fetch_trial(
        experiment: ExperimentReference, trial: TrialReference, user: str
) -> Optional[Tuple[Dict[str, Any], Tuple]]:
    """
    Fetches a trial and returns trial information dictionary.
    Args:
        experiment: The experiment to fetch information from
        trial: The trial to fetch information from

    Returns:
    """
    fetch_args = FetchTrialRawArgs(user=user, trial_id=trial.id)
    try:
        trial_info = _fetch_trial_raw(fetch_args)
    except APIException:
        print('WARN: got no checkpoint for trial {}'.format(trial.id), file=sys.stderr)
        return None

    steps = trial_info['steps']
    if not steps:
        print('WARN: got no checkpoint for trial {}'.format(trial.id), file=sys.stderr)
        return None
    last_step = steps[-1]
    if last_step['state'] != 'COMPLETED':
        print('WARN: got no checkpoint for trial {}'.format(trial.id), file=sys.stderr)
        return None

    experiment_id = experiment.id
    experiment_data = {
        'experiment_id': experiment_id,
        'trial_id': trial.id,
        'created_checkpoint_uuid': get_dict_recursive(last_step, ['checkpoint', 'uuid']),
        'experiment_seed': experiment.get_config()['reproducibility']['experiment_seed']
    }

    # Hyperparameters
    try:
        hparams = trial_info['hparams']
        missing_hparams, ignored_hparams = insert_dict(hparams, experiment_data, HPARAM_KEYS)
    except KeyError:
        print(f'got no hparams for trial {trial.id} in experiment {experiment_id}:\n', trial_info.keys())
        return None

    # Metrics
    metrics = get_dict_recursive(last_step, ['validation', 'metrics', 'validation_metrics'])
    # metrics = last_step.get('validation', {}).get('metrics', {}).get('validation_metrics')
    if metrics is None:
        print('WARN: ignoring trial {}'.format(trial.id))
        return None

    missing_metrics, ignored_metrics = insert_dict(metrics, experiment_data, METRIC_KEYS)

    return experiment_data, (missing_hparams, ignored_hparams, missing_metrics, ignored_metrics)


def fetch_trial_with_det_api(
        experiment: ExperimentReference,
        trial: TrialReference
) -> Optional[Tuple[Dict[str, Any], Tuple]]:
    """
    THIS FUNCTION IS DEPRECATED AS IT TAKES FOREVER.
    Fetches a trial and returns trial information dictionary.
    Args:
        experiment: The experiment to fetch information from
        trial: The trial to fetch information from

    Returns:

    """
    try:
        checkpoint = trial.select_checkpoint(latest=True)
    except AssertionError:
        print('No checkpoint for trial {} -> skipping.'.format(trial.id), file=sys.stderr)
        return

    experiment_id = experiment.id
    experiment_data = {
        'experiment_id': experiment_id,
        'trial_id': trial.id,
        'created_checkpoint_uuid': checkpoint.uuid,
        'experiment_seed': experiment.get_config()['reproducibility']['experiment_seed']
    }

    # Hyperparameters
    hparams = checkpoint.training.hparams
    missing_hparams, ignored_hparams = insert_dict(hparams, experiment_data, HPARAM_KEYS)

    # Metrics
    metrics = checkpoint.training.validation_metrics['avgMetrics']
    missing_metrics, ignored_metrics = insert_dict(metrics, experiment_data, METRIC_KEYS)

    return experiment_data, (missing_hparams, ignored_hparams, missing_metrics, ignored_metrics)


def fetch_dataframe(
        experiment_ids: List[int],
        determined_client: Determined,
        read_cache: bool = False,
        threads: int = 2,
        show_progress: bool = False,
) -> Tuple[pd.DataFrame, Tuple]:
    """
    Fetches the given experiment ids from the determined master and returns them in a pandas dataframe.
    Args:
        experiment_ids: A list of experiment ids
        determined_client: The determined client to fetch experiment information from
        read_cache: Whether to read from csv data
        threads: The number of threads to fetch trial information. Defaults to 4.

    Returns:
        Pandas Dataframe containing the experiment information of the given experiment ids and information about missing
        and ignored hparams and metrics.
    """
    cached_dataframe = None
    cached_experiment_ids = []
    if read_cache and DATA_PATH.is_file():
        cached_dataframe = pd.read_csv(DATA_PATH)
        cached_experiment_ids = list(cached_dataframe['experiment_id'].unique())
        unused_experiment_ids = filter(lambda exp_id: exp_id not in experiment_ids, cached_experiment_ids)
        # remove unused experiment ids
        for unused_exp_id in unused_experiment_ids:
            cached_dataframe = cached_dataframe[cached_dataframe['experiment_id'] != unused_exp_id]
        cached_experiment_ids = list(cached_dataframe['experiment_id'].unique())

    data = {}
    missing_hparams = set()
    ignored_hparams = set()
    missing_metrics = set()
    ignored_metrics = set()

    def _progress_function(x):
        return x
    if show_progress:
        try:
            import tqdm
            _progress_function = tqdm.tqdm
        except ModuleNotFoundError:
            print('WARN: Failed to import tqdm. Install tqdm to enable progress monitoring', file=sys.stderr)

    user = get_credentials()['username']
    for experiment_id in experiment_ids:
        # don't fetch again if experiment is cached
        if experiment_id in cached_experiment_ids:
            continue
        experiment = determined_client.get_experiment(experiment_id)

        # NOTE: was a workaround for experiments with trials > 100
        # trials_a = experiment.get_trials(
        #     sort_by=determined.common.experimental.trial.TrialSortBy.ID,
        #     order_by=determined.common.experimental.trial.TrialOrderBy.ASCENDING)
        # trials_b = experiment.get_trials(
        #     sort_by=determined.common.experimental.trial.TrialSortBy.ID,
        #     order_by=determined.common.experimental.trial.TrialOrderBy.DESC)
        # trials = trials_a + trials_b[:28]

        trials = experiment.get_trials()

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []

            for trial in trials:
                futures.append(executor.submit(fetch_trial, experiment=experiment, trial=trial, user=user))

            for future in _progress_function(futures):
                result = future.result()
                if result is None:
                    continue
                experiment_data, missing_ignored = result
                mh, ih, mm, im = missing_ignored
                missing_hparams.update(mh)
                ignored_hparams.update(ih)
                missing_metrics.update(mm)
                ignored_metrics.update(im)

                for key, value in experiment_data.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)

    result_dataframe = pd.DataFrame(data)

    if cached_dataframe is not None:
        result_dataframe = pd.concat([cached_dataframe, result_dataframe], ignore_index=True)

    return result_dataframe, (missing_hparams, ignored_hparams, missing_metrics, ignored_metrics)


def warn_missing_ignored(missing_hparams, ignored_hparams, missing_metrics, ignored_metrics):
    if missing_hparams:
        print(
            'The following hparams could not be found in experiment: [{}]'.format(', '.join(missing_hparams)),
            file=sys.stderr
        )
    if ignored_hparams:
        print('The following hparams were not copied: [{}]'.format(', '.join(ignored_hparams)), file=sys.stderr)
    if missing_metrics:
        print(
            'The following metrics could not be found in experiment: [{}]'.format(', '.join(missing_metrics)),
            file=sys.stderr
        )
    if ignored_metrics:
        print('The following metrics were not copied: [{}]'.format(', '.join(ignored_metrics)), file=sys.stderr)


def main():
    args = get_args()
    d = get_determined_client()

    experiment_ids = args.experiment_id

    new_dataframe, missing_ignored = fetch_dataframe(experiment_ids, d, read_cache=True, show_progress=True)
    warn_missing_ignored(*missing_ignored)

    if DATA_PATH.is_file() and args.merge:
        old_dataframe = pd.read_csv(DATA_PATH)
        # remove experiments from old dataframe which are newly fetched
        for experiment_id in experiment_ids:
            old_dataframe = old_dataframe[old_dataframe['experiment_id'] != experiment_id]
        result_dataframe = pd.concat([old_dataframe, new_dataframe], ignore_index=True)
    else:
        result_dataframe = new_dataframe

    result_dataframe.to_csv(DATA_PATH, index=False)


if __name__ == '__main__':
    main()
