import os
import sys
import logging
from pathlib import Path


def init_auto_log(name: str):
    # path
    experiment = 'experiment{:0>5}'.format(os.environ.get('DET_EXPERIMENT_ID', '_unknown'))
    log_dir = Path('/logs') / name / experiment
    os.makedirs(log_dir, exist_ok=True)

    # stdout / stderr
    stdout_path = log_dir / 'out'
    sys.stdout = open(stdout_path, 'w')
    stderr_path = log_dir / 'err'
    sys.stderr = open(stderr_path, 'w')

    # exception handler
    log_path = log_dir / 'trace'
    outfile = open(log_path, 'w')
    handler = logging.StreamHandler(stream=outfile)

    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _handle_exception
