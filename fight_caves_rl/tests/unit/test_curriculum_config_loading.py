from fight_caves_rl.curriculum.registry import build_curriculum, load_curriculum_config


def test_curriculum_disabled_returns_no_overrides():
    config = load_curriculum_config("curriculum_disabled_v0")
    curriculum = build_curriculum("curriculum_disabled_v0")

    assert config.enabled is False
    assert config.mode == "disabled"
    assert curriculum.reset_overrides(slot_index=0, episode_index=0) == {}


def test_wave_progression_curriculum_applies_expected_schedule():
    config = load_curriculum_config("curriculum_wave_progression_v0")
    curriculum = build_curriculum("curriculum_wave_progression_v0")

    assert config.enabled is True
    assert config.mode == "wave_progression"
    assert len(config.schedule) == 3
    assert curriculum.reset_overrides(slot_index=0, episode_index=0) == {"start_wave": 1}
    assert curriculum.reset_overrides(slot_index=0, episode_index=7) == {"start_wave": 1}
    assert curriculum.reset_overrides(slot_index=0, episode_index=8) == {"start_wave": 8}
    assert curriculum.reset_overrides(slot_index=0, episode_index=15) == {"start_wave": 8}
    assert curriculum.reset_overrides(slot_index=0, episode_index=16) == {"start_wave": 31}


def test_wave_progression_v2_curriculum_applies_mixed_schedule():
    config = load_curriculum_config("curriculum_wave_progression_v2")
    curriculum = build_curriculum("curriculum_wave_progression_v2")

    assert config.enabled is True
    assert config.mode == "wave_progression"
    assert len(config.schedule) == 6
    assert curriculum.reset_overrides(slot_index=0, episode_index=0) == {"start_wave": 1}
    assert curriculum.reset_overrides(slot_index=0, episode_index=8) == {"start_wave": 16}
    assert curriculum.reset_overrides(slot_index=0, episode_index=16) == {"start_wave": 31}
    assert curriculum.reset_overrides(slot_index=0, episode_index=24) == {"start_wave": 62}
    assert curriculum.reset_overrides(slot_index=0, episode_index=40) == {"start_wave": 1}
    assert curriculum.reset_overrides(slot_index=0, episode_index=41) == {"start_wave": 62}
    assert curriculum.reset_overrides(slot_index=0, episode_index=42) == {"start_wave": 1}
