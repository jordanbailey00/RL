from fight_caves_rl.manifests.run_manifest import build_bootstrap_manifest
from fight_caves_rl.utils.config import load_bootstrap_config


def test_bootstrap_manifest_contains_expected_fields():
    manifest = build_bootstrap_manifest(load_bootstrap_config({}))
    payload = manifest.to_dict()

    assert payload["rl_repo"].endswith("/home/jordan/code/RL")
    assert payload["sim_repo"].endswith("/home/jordan/code/fight-caves-RL")
    assert payload["rsps_repo"].endswith("/home/jordan/code/RSPS")
    assert payload["python_baseline"] == "3.11"
    assert payload["pufferlib_distribution"] == "pufferlib-core"
    assert payload["pufferlib_version"] == "3.0.17"
    assert payload["wandb_mode"] == "offline"
    assert payload["created_at"]
