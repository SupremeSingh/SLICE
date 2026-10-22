"""Initialize a native Duke controller on the current login host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys

from .jobs import JobStore
from .slurm import settings, validate_settings


def init_cluster(path, *, state_dir=None, checkout=None):
    """Create configuration once, with a private node-local SQLite ledger.

    /var/tmp is working storage, not a durable backup. The shared configuration
    pins the login host and requires the ledger to exist on subsequent runs.
    """
    path = Path(path).absolute()
    if path.exists():
        return settings(path)
    checkout = Path(checkout or Path.cwd()).resolve()
    source = checkout / 'src'
    if not (source / 'slice_probe').is_dir():
        raise ValueError('Run init-cluster from the SLICE checkout containing src/slice_probe')
    state = Path(state_dir or f'/var/tmp/slice-controller-{os.getuid()}').absolute()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.is_symlink() or state.stat().st_uid != os.getuid():
        raise ValueError('Controller state directory must be owned by you and not a symlink')
    state.chmod(0o700)
    config = validate_settings(dict(
        database=str(state / 'jobs.sqlite'),
        registry=str(checkout.parent / 'slice-critics'),
        cluster_registry=str(checkout.parent / 'slice-critics'),
        cluster_root=str(checkout.parent / 'slice-service'),
        cluster_source=str(source), cluster_python=sys.executable,
        partition='compsci-gpu', account=None, gres='gpu:a5000:1',
        cpus=8, memory_gb=64, max_total_minutes=1440, device='auto',
        controller_host=socket.gethostname(), require_existing_database=True,
    ))
    JobStore(config['database'])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing configuration, even during concurrent setup.
    with path.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    return config
