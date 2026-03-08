from pathlib import Path

from fight_caves_rl.utils.paths import repo_root


def test_required_pr2_docs_exist_and_contain_anchor_terms():
    root = repo_root()
    expected = {
        "docs/rl_integration_contract.md": ("FightCaveEpisodeInitializer.reset", "headless_observation_v1"),
        "docs/bridge_contract.md": (":game:headlessDistZip", "HeadlessMain.bootstrap"),
        "docs/observation_mapping.md": ("headless_observation_v1", "visible_index"),
        "docs/action_mapping.md": ("headless_action_v1", "AttackVisibleNpc"),
        "docs/reward_configs.md": ("reward_sparse_v0", "curriculum_wave_progression_v0"),
        "docs/hotpath_map.md": ("Mode A", "Mode C"),
        "docs/performance_plan.md": ("1,000,000 env steps/sec", "official_profile_v0"),
    }

    for relative_path, snippets in expected.items():
        path = root / relative_path
        assert path.is_file(), relative_path
        text = path.read_text()
        for snippet in snippets:
            assert snippet in text, f"{snippet!r} missing from {relative_path}"
