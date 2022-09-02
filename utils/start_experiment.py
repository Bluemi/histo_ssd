import argparse
from pprint import pprint

from determined.common.experimental import Determined
from ruamel.yaml import YAML


VARIABLES_KEY = 'variables'
MODEL_DIR = 'src'
yaml = YAML(typ='safe')


def parse_args():
    parser = argparse.ArgumentParser(description='Start and patch experiments')
    parser.add_argument('config', type=argparse.FileType('r'))
    parser.add_argument('patches', type=argparse.FileType('r'), nargs='*')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--dry', action='store_true')

    return parser.parse_args()


def update_dict_recursive(target_dict, source_dict, verbose=False):
    for key, value in source_dict.items():
        if key not in target_dict:
            target_dict[key] = value
            print('patching "{}": {}'.format(key, value))
        elif isinstance(target_dict[key], dict):
            assert isinstance(source_dict[key], dict)
            update_dict_recursive(target_dict[key], value, verbose=False)
            if verbose:
                print('patching "{}": {}'.format(key, value))
        elif isinstance(target_dict[key], list):
            assert isinstance(source_dict[key], list)
            raise NotImplementedError('cant update lists')
        else:
            old_value = target_dict.get(key)
            target_dict[key] = value
            if verbose:
                print('patching "{}": {} -> {}'.format(key, old_value, value))


def main():
    args = parse_args()
    experiment_config = yaml.load(args.config)

    variables = {
        'description': {
            'augmentation': 'no augmentation',
            'action': 'test',
        }
    }
    for patch in args.patches:
        patch_data = yaml.load(patch)

        # save variables
        if VARIABLES_KEY in patch_data:
            update_dict_recursive(variables, patch_data[VARIABLES_KEY], verbose=args.verbose)
            del patch_data[VARIABLES_KEY]

        # patch config
        update_dict_recursive(experiment_config, patch_data, verbose=args.verbose)

    for config_key, update_dict in variables.items():
        old_value = experiment_config[config_key]
        experiment_config[config_key] = experiment_config[config_key].format(**update_dict)
        if args.verbose:
            print('patched "{}": {} -> {}'.format(config_key, old_value, experiment_config[config_key]))

    if args.verbose >= 2:
        pprint(experiment_config)

    if args.dry:
        pprint(experiment_config)
    else:
        d = Determined(master='https://dt1.f4.htw-berlin.de:8443')
        experiment_ref = d.create_experiment(experiment_config, MODEL_DIR)
        print(experiment_ref.id)


if __name__ == '__main__':
    main()
