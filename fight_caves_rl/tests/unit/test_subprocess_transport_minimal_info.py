from __future__ import annotations

import numpy as np

from fight_caves_rl.envs.subprocess_vector_env import (
    _partition_worker_env_counts,
    _serialize_transition,
)
from fight_caves_rl.envs.shared_memory_transport import INFO_PAYLOAD_MODE_MINIMAL


def test_pipe_transport_omits_minimal_infos_from_payload():
    transition = (
        np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
        np.asarray([0.5, 1.5], dtype=np.float32),
        np.asarray([False, True], dtype=np.bool_),
        np.asarray([False, False], dtype=np.bool_),
        np.asarray([0, 0], dtype=np.int32),
        [{}, {}],
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([True, True], dtype=np.bool_),
    )

    payload = _serialize_transition(transition, transport_worker=None)

    assert payload["info_payload_mode"] == INFO_PAYLOAD_MODE_MINIMAL
    assert "infos" not in payload


def test_partition_worker_env_counts_balances_envs_across_workers():
    assert _partition_worker_env_counts(env_count=4, worker_count=2) == (2, 2)
    assert _partition_worker_env_counts(env_count=5, worker_count=2) == (3, 2)
    assert _partition_worker_env_counts(env_count=3, worker_count=5) == (1, 1, 1)
