from fight_caves_rl.utils.config import load_bootstrap_config


def test_load_bootstrap_config_defaults_to_workspace_paths():
    config = load_bootstrap_config({})

    assert config.rl_repo.as_posix().endswith("/home/jordan/code/RL")
    assert config.sim_repo.as_posix().endswith("/home/jordan/code/fight-caves-RL")
    assert config.rsps_repo.as_posix().endswith("/home/jordan/code/RSPS")
    assert config.python_baseline == "3.11"
    assert config.pufferlib_version == "3.0.0"
    assert config.wandb_mode == "offline"


def test_load_bootstrap_config_honors_environment_overrides():
    config = load_bootstrap_config(
        {
            "RL_REPO": "/tmp/rl",
            "FIGHT_CAVES_RL_REPO": "/tmp/sim",
            "RSPS_REPO": "/tmp/rsps",
            "PYTHON_BASELINE": "3.12",
            "PUFFERLIB_VERSION": "9.9.9",
            "WANDB_PROJECT": "custom-project",
            "WANDB_MODE": "online",
        }
    )

    assert config.rl_repo.as_posix() == "/tmp/rl"
    assert config.sim_repo.as_posix() == "/tmp/sim"
    assert config.rsps_repo.as_posix() == "/tmp/rsps"
    assert config.python_baseline == "3.12"
    assert config.pufferlib_version == "9.9.9"
    assert config.wandb_project == "custom-project"
    assert config.wandb_mode == "online"

