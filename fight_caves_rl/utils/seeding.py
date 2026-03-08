from __future__ import annotations

from random import Random

from fight_caves_rl.replay.seed_packs import SeedPack, resolve_seed_pack


def canonical_seed_sequence(pack: str | SeedPack) -> tuple[int, ...]:
    seed_pack = resolve_seed_pack(pack) if isinstance(pack, str) else pack
    return tuple(int(seed) for seed in seed_pack.seeds)


def deterministic_python_rng(seed: int) -> Random:
    return Random(int(seed))
