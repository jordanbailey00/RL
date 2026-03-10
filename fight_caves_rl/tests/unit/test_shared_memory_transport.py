from __future__ import annotations

import numpy as np

from fight_caves_rl.envs.shared_memory_transport import (
    SHARED_MEMORY_TRANSPORT_MODE,
    SharedMemoryTransportParent,
    SharedMemoryTransportWorker,
)


def test_shared_memory_transport_round_trip():
    parent = SharedMemoryTransportParent(env_count=2, action_dim=6, observation_dim=4)
    worker = SharedMemoryTransportWorker.attach(parent.spec().to_payload())
    try:
        actions = np.arange(12, dtype=np.int32).reshape(2, 6)
        parent.write_actions(actions)
        np.testing.assert_array_equal(worker.read_actions(), actions)

        transition = (
            np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
            np.asarray([0.5, 1.5], dtype=np.float32),
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([False, False], dtype=np.bool_),
            np.asarray([0, 0], dtype=np.int32),
            [{"slot_index": 0}, {"slot_index": 1}],
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([True, True], dtype=np.bool_),
        )
        payload = worker.publish_transition(transition)
        materialized = parent.materialize_transition(payload)

        assert payload["transport_mode"] == SHARED_MEMORY_TRANSPORT_MODE
        np.testing.assert_array_equal(materialized["observations"], transition[0])
        np.testing.assert_array_equal(materialized["rewards"], transition[1])
        np.testing.assert_array_equal(materialized["terminals"], transition[2])
        np.testing.assert_array_equal(materialized["truncations"], transition[3])
        np.testing.assert_array_equal(materialized["teacher_actions"], transition[4])
        assert materialized["infos"] == transition[5]
        np.testing.assert_array_equal(materialized["agent_ids"], transition[6])
        np.testing.assert_array_equal(materialized["masks"], transition[7])
    finally:
        worker.close()
        parent.close(unlink=True)
