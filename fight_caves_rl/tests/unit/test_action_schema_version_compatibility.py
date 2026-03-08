import pytest

from fight_caves_rl.envs.action_mapping import normalize_action
from fight_caves_rl.envs.schema import HEADLESS_ACTION_DEFINITIONS, HEADLESS_PROTECTION_PRAYER_IDS


def test_action_normalizer_matches_contract_ids_and_shapes():
    assert [definition.action_id for definition in HEADLESS_ACTION_DEFINITIONS] == list(range(7))

    walk = normalize_action({"action_id": 1, "tile": {"x": 10, "y": 20}})
    attack = normalize_action({"name": "attack_visible_npc", "visible_npc_index": 3})
    prayer = normalize_action({"action_id": 3, "prayer": "protect_from_magic"})

    assert walk.name == "walk_to_tile"
    assert (walk.tile.x, walk.tile.y, walk.tile.level) == (10, 20, 0)
    assert attack.visible_npc_index == 3
    assert prayer.prayer in HEADLESS_PROTECTION_PRAYER_IDS


def test_action_normalizer_rejects_contract_breaking_payloads():
    with pytest.raises(ValueError):
        normalize_action({"action_id": 1})

    with pytest.raises(ValueError):
        normalize_action({"action_id": 2})

    with pytest.raises(ValueError):
        normalize_action({"action_id": 3, "prayer": "smite"})

    with pytest.raises(ValueError):
        normalize_action({"action_id": 0, "visible_npc_index": 1})
