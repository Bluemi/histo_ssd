import os
import logging
from pathlib import Path
import sys


def init_auto_log(name: str):
    # path
    logger = logging.getLogger(__name__)
    experiment = 'experiment{:0>5}'.format(os.environ.get('DET_EXPERIMENT_ID', '_unknown'))
    log_dir = Path('/logs') / name / experiment
    os.makedirs(log_dir, exist_ok=True)

    # stdout / stderr
    stdout_path = log_dir / 'stdout'
    sys.stdout = open(stdout_path, 'w')
    stderr_path = log_dir / 'stderr'
    sys.stderr = open(stderr_path, 'w')
    log_path = log_dir / 'stacktrace'

    # exception handler
    outfile = open(log_path, 'w')
    handler = logging.StreamHandler(stream=outfile)
    logger.addHandler(handler)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _handle_exception
