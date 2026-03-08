from fight_caves_rl.utils.config import load_bootstrap_config


def test_load_bootstrap_config_defaults_to_workspace_paths():
    config = load_bootstrap_config({})

    assert config.rl_repo.as_posix().endswith("/home/jordan/code/RL")
    assert config.sim_repo.as_posix().endswith("/home/jordan/code/fight-caves-RL")
    assert config.rsps_repo.as_posix().endswith("/home/jordan/code/RSPS")
    assert config.python_baseline == "3.11"
    assert config.pufferlib_distribution == "pufferlib-core"
    assert config.pufferlib_version == "3.0.17"
    assert config.wandb_group == "smoke"
    assert config.wandb_mode == "offline"
    assert config.wandb_resume == "allow"
    assert config.wandb_run_prefix == "fc-rl"
    assert config.wandb_tags == ("fight-caves", "rl")
    assert config.wandb_dir.as_posix().endswith("/home/jordan/code/RL/artifacts/wandb")
    assert config.wandb_data_dir.as_posix().endswith("/home/jordan/code/RL/artifacts/wandb-data")
    assert config.wandb_cache_dir.as_posix().endswith("/home/jordan/code/RL/artifacts/wandb-cache")


def test_load_bootstrap_config_honors_environment_overrides():
    config = load_bootstrap_config(
        {
            "RL_REPO": "/tmp/rl",
            "FIGHT_CAVES_RL_REPO": "/tmp/sim",
            "RSPS_REPO": "/tmp/rsps",
            "PYTHON_BASELINE": "3.12",
            "PUFFERLIB_DISTRIBUTION": "custom-pufferlib",
            "PUFFERLIB_VERSION": "9.9.9",
            "WANDB_PROJECT": "custom-project",
            "WANDB_ENTITY": "custom-entity",
            "WANDB_GROUP": "custom-group",
            "WANDB_MODE": "online",
            "WANDB_RESUME": "must",
            "WANDB_RUN_PREFIX": "custom-prefix",
            "WANDB_TAGS": "alpha,beta",
            "WANDB_DIR": "/tmp/wandb",
            "WANDB_DATA_DIR": "/tmp/wandb-data",
            "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        }
    )

    assert config.rl_repo.as_posix() == "/tmp/rl"
    assert config.sim_repo.as_posix() == "/tmp/sim"
    assert config.rsps_repo.as_posix() == "/tmp/rsps"
    assert config.python_baseline == "3.12"
    assert config.pufferlib_distribution == "custom-pufferlib"
    assert config.pufferlib_version == "9.9.9"
    assert config.wandb_project == "custom-project"
    assert config.wandb_entity == "custom-entity"
    assert config.wandb_group == "custom-group"
    assert config.wandb_mode == "online"
    assert config.wandb_resume == "must"
    assert config.wandb_run_prefix == "custom-prefix"
    assert config.wandb_tags == ("alpha", "beta")
    assert config.wandb_dir.as_posix() == "/tmp/wandb"
    assert config.wandb_data_dir.as_posix() == "/tmp/wandb-data"
    assert config.wandb_cache_dir.as_posix() == "/tmp/wandb-cache"
