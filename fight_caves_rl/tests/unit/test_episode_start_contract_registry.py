from fight_caves_rl.envs.schema import FIGHT_CAVE_EPISODE_START_CONTRACT


def test_episode_start_contract_matches_verified_sim_defaults():
    contract = FIGHT_CAVE_EPISODE_START_CONTRACT

    assert contract.identity.contract_id == "fight_cave_episode_start_v1"
    assert contract.start_wave_min == 1
    assert contract.start_wave_max == 63
    assert contract.default_start_wave == 1
    assert contract.default_ammo == 1000
    assert contract.default_prayer_potions == 8
    assert contract.default_sharks == 20
    assert contract.run_energy_percent == 100
    assert contract.run_toggle_on is True
    assert contract.xp_gain_blocked is True


def test_episode_start_contract_keeps_fixed_levels_and_loadout():
    contract = FIGHT_CAVE_EPISODE_START_CONTRACT

    assert contract.fixed_levels == (
        ("Attack", 1),
        ("Strength", 1),
        ("Defence", 70),
        ("Constitution", 700),
        ("Ranged", 70),
        ("Prayer", 43),
        ("Magic", 1),
    )
    assert contract.equipment == (
        "coif",
        "rune_crossbow",
        "black_dragonhide_body",
        "black_dragonhide_chaps",
        "black_dragonhide_vambraces",
        "snakeskin_boots",
        "adamant_bolts",
    )
    assert contract.inventory_item_ids == ("prayer_potion_4", "shark")
    assert contract.start_wave_invocation == "fightCave.startWave(player, startWave, start = false)"
