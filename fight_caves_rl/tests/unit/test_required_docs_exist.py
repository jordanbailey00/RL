from pathlib import Path

from fight_caves_rl.utils.paths import repo_root, workspace_root


def test_required_phase0_docs_exist_and_contain_anchor_terms():
    root = repo_root()
    workspace = workspace_root()
    expected = {
        workspace / "pivot_plan.md": ("fight-caves-demo-lite", "mechanics parity"),
        workspace / "pivot_implementation_plan.md": ("fight_caves_rl/contracts", "terminal codes"),
        root / "RLspec.md": ("RSPS-backed headed demo/replay path", "WSL/Linux"),
        root / "README.md": ("pivot_plan.md", "rsps_headed"),
        root / "docs/rl_integration_contract.md": (
            "RSPS-backed headed demo",
            "WSL/Linux is the canonical environment",
        ),
        root / "docs/bridge_contract.md": (
            "must not reconstruct structured semantics",
            "WSL/Linux is the canonical environment",
        ),
        root / "docs/default_backend_selection.md": ("rsps_headed", "fight-caves-demo-lite"),
        root / "docs/observation_mapping.md": ("headless_observation_v1", "visible_index"),
        root / "docs/action_mapping.md": ("headless_action_v1", "AttackVisibleNpc"),
        root / "docs/eval_and_replay.md": ("replay_pack_v0", "replay_step_cadence"),
        root / "docs/reward_configs.md": (
            "correct_jad_prayer_on_resolve",
            "tick_penalty_base",
        ),
    }

    for path, snippets in expected.items():
        assert path.is_file(), path
        text = path.read_text()
        for snippet in snippets:
            assert snippet in text, f"{snippet!r} missing from {path}"
