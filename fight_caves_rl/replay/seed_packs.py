from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.envs.schema import VersionedContract


@dataclass(frozen=True)
class SeedPack:
    identity: VersionedContract
    seeds: tuple[int, ...]
    description: str
    source_ref: str


BOOTSTRAP_SMOKE_SEED_PACK = SeedPack(
    identity=VersionedContract(
        contract_id="bootstrap_smoke",
        version=0,
        compatibility_policy="append_only_or_bump",
    ),
    seeds=(11_001, 22_002),
    description="Thin deterministic smoke seeds aligned to the first parity harness baselines in the sim repo.",
    source_ref="fight-caves-RL headless parity harness seeds 11001 and 22002",
)

PARITY_REFERENCE_SEED_PACK = SeedPack(
    identity=VersionedContract(
        contract_id="parity_reference_v0",
        version=0,
        compatibility_policy="append_only_or_bump",
    ),
    seeds=(11_001, 33_003, 44_004),
    description="Seed pack aligned to the current single-wave, Jad healer, and Tz-Kek split parity scenarios in the sim repo.",
    source_ref="fight-caves-RL parity harness seeds 11001, 33003, and 44004",
)


SEED_PACKS = {
    BOOTSTRAP_SMOKE_SEED_PACK.identity.contract_id: BOOTSTRAP_SMOKE_SEED_PACK,
    PARITY_REFERENCE_SEED_PACK.identity.contract_id: PARITY_REFERENCE_SEED_PACK,
}


def resolve_seed_pack(pack_id: str) -> SeedPack:
    try:
        return SEED_PACKS[pack_id]
    except KeyError as exc:
        raise KeyError(f"Unknown seed pack id: {pack_id!r}") from exc


def seed_pack_ids() -> tuple[str, ...]:
    return tuple(SEED_PACKS.keys())
