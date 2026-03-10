from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fight_caves_rl.bridge.contracts import (
    BridgeHandshake,
    HeadlessBootstrapConfig,
    HeadlessEpisodeConfig,
    HeadlessPlayerConfig,
    HeadlessRuntimePaths,
    StepSnapshot,
)
from fight_caves_rl.bridge.errors import BridgeError, BridgeJVMStateError
from fight_caves_rl.bridge.launcher import (
    assert_sim_runtime_ready,
    build_bridge_handshake,
    build_headless_settings_overrides,
    discover_headless_runtime_paths,
)
from fight_caves_rl.envs.action_mapping import NormalizedAction, TileCoordinates, normalize_action
from fight_caves_rl.envs.observation_views import coerce_flat_observation_row
from fight_caves_rl.envs.observation_mapping import validate_observation_contract
from fight_caves_rl.utils.java_runtime import resolve_jvm_library_path


@dataclass
class _JVMContext:
    jpype: Any
    classpath: Path
    user_dir: Path
    classes: dict[str, Any]


_JVM_CONTEXT: _JVMContext | None = None
QUIET_LOGBACK_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "logging" / "headless_quiet_logback.xml"
)


class HeadlessDebugClient:
    def __init__(
        self,
        paths: HeadlessRuntimePaths,
        bootstrap: HeadlessBootstrapConfig | None = None,
    ) -> None:
        self.paths = paths
        self.bootstrap = bootstrap or HeadlessBootstrapConfig()
        self.handshake: BridgeHandshake = build_bridge_handshake(paths)
        self._jvm = _ensure_jvm(paths.headless_jar, paths.launch_cwd)
        self._runtime: Any | None = None
        self._closed = False

    @classmethod
    def create(
        cls,
        sim_repo: Path | None = None,
        bootstrap: HeadlessBootstrapConfig | None = None,
    ) -> "HeadlessDebugClient":
        paths = discover_headless_runtime_paths(sim_repo=sim_repo)
        assert_sim_runtime_ready(paths)
        client = cls(paths=paths, bootstrap=bootstrap)
        client.bootstrap_runtime()
        return client

    def bootstrap_runtime(self) -> None:
        if self._runtime is not None:
            return
        overrides = build_headless_settings_overrides(self.paths, self.bootstrap)
        override_map = self._jvm["HashMap"]()
        for key, value in overrides.items():
            override_map.put(key, value)

        self._runtime = self._jvm["HeadlessMain"].INSTANCE.bootstrap(
            self.bootstrap.load_content_scripts,
            self.bootstrap.start_world,
            self.bootstrap.install_shutdown_hook,
            override_map,
        )

    def create_player_slot(self, config: HeadlessPlayerConfig) -> Any:
        self._ensure_runtime_open()
        account_manager = self._resolve_account_manager()
        player = self._construct_player(config)
        if not account_manager.setup(player, None, 0, True):
            raise BridgeError(
                f"Failed to setup headless player {config.account_name!r}."
            )
        player.set("creation", -1)
        player.set("skip_level_up", True)
        account_manager.spawn(player, None)

        viewport = player.getViewport()
        if viewport is None:
            raise BridgeError(
                f"Headless player {config.account_name!r} does not have a viewport."
            )
        if hasattr(viewport, "setLoaded"):
            viewport.setLoaded(True)
        else:
            viewport.loaded = True
        return player

    def reset_episode(self, player: Any, config: HeadlessEpisodeConfig) -> dict[str, Any]:
        self._ensure_runtime_open()
        episode_config = self._jvm["FightCaveEpisodeConfig"](
            int(config.seed),
            int(config.start_wave),
            int(config.ammo),
            int(config.prayer_potions),
            int(config.sharks),
        )
        state = self._runtime.resetFightCaveEpisode(player, episode_config)
        return {
            "seed": int(state.getSeed()),
            "wave": int(state.getWave()),
            "rotation": int(state.getRotation()),
            "remaining": int(state.getRemaining()),
            "instance_id": int(state.getInstanceId()),
            "player_tile": self._tile_from_id(
                getattr(state, "getPlayerTile-aANnmZU")()
            ),
        }

    def observe(self, player: Any, include_future_leakage: bool = False) -> dict[str, Any]:
        self._ensure_runtime_open()
        observation = self.observe_jvm(player, include_future_leakage=include_future_leakage)
        return pythonize_observation(observation)

    def observe_flat(self, player: Any) -> np.ndarray:
        self._ensure_runtime_open()
        observation = self.observe_flat_jvm(player)
        return flat_observation_to_numpy(observation)

    def visible_targets(self, player: Any) -> list[dict[str, Any]]:
        self._ensure_runtime_open()
        targets = []
        for target in self._runtime.visibleFightCaveNpcTargets(player):
            targets.append(
                {
                    "visible_index": int(target.getVisibleIndex()),
                    "npc_index": int(target.getNpcIndex()),
                    "id": str(target.getId()),
                    "tile": self._tile_from_id(
                        getattr(target, "getTile-aANnmZU")()
                    ),
                }
            )
        return targets

    def apply_action(self, player: Any, action: int | str | dict[str, object] | NormalizedAction) -> dict[str, Any]:
        self._ensure_runtime_open()
        result = self.apply_action_jvm(player, action)
        return pythonize_action_result(result)

    def step_once(
        self,
        player: Any,
        action: int | str | dict[str, object] | NormalizedAction,
        ticks_after: int = 1,
        include_future_leakage: bool = False,
    ) -> StepSnapshot:
        action_result = self.apply_action(player, action)
        if ticks_after > 0:
            self._runtime.tick(int(ticks_after))
        observation = self.observe(player, include_future_leakage=include_future_leakage)
        return StepSnapshot(
            observation=observation,
            action_result=action_result,
            visible_targets=self.visible_targets(player),
        )

    def tick(self, times: int = 1) -> None:
        self._ensure_runtime_open()
        if int(times) < 0:
            raise BridgeError(f"tick times must be >= 0, got {times}.")
        if int(times) == 0:
            return
        self._runtime.tick(int(times))

    def build_action(self, action: int | str | dict[str, object] | NormalizedAction) -> Any:
        self._ensure_runtime_open()
        return self._build_jvm_action(normalize_action(action))

    def apply_action_jvm(
        self,
        player: Any,
        action: int | str | dict[str, object] | NormalizedAction,
    ) -> Any:
        self._ensure_runtime_open()
        return self._runtime.applyFightCaveAction(player, self.build_action(action))

    def observe_jvm(self, player: Any, include_future_leakage: bool = False) -> Any:
        self._ensure_runtime_open()
        return self._runtime.observeFightCave(player, include_future_leakage)

    def observe_flat_jvm(self, player: Any) -> Any:
        self._ensure_runtime_open()
        return self._runtime.observeFightCaveFlat(player)

    def run_action_trace(
        self,
        player: Any,
        actions: list[int | str | dict[str, object] | NormalizedAction],
        *,
        ticks_after: int = 1,
        observe_every: int = 0,
        include_future_leakage: bool = False,
    ) -> dict[str, Any]:
        self._ensure_runtime_open()
        if int(ticks_after) < 0:
            raise BridgeError(f"ticks_after must be >= 0, got {ticks_after}.")
        if int(observe_every) < 0:
            raise BridgeError(f"observe_every must be >= 0, got {observe_every}.")

        replay_steps = self._jvm["ArrayList"]()
        for action in actions:
            replay_steps.add(
                self._jvm["HeadlessReplayStep"](
                    self.build_action(action),
                    int(ticks_after),
                )
            )

        batch = self._jvm["HeadlessBatchSteppingKt"].runFightCaveBatch(
            self._runtime,
            player,
            replay_steps,
            int(observe_every),
            bool(include_future_leakage),
        )
        final_observation = pythonize_observation(batch.getFinalObservation())
        return {
            "steps": int(batch.getSteps()),
            "ticks_advanced": int(batch.getTicksAdvanced()),
            "action_applied_count": int(batch.getActionAppliedCount()),
            "elapsed_nanos": int(batch.getElapsedNanos()),
            "steps_per_second": float(batch.getStepsPerSecond()),
            "ticks_per_second": float(batch.getTicksPerSecond()),
            "final_observation": final_observation,
        }

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None
        self._closed = True

    def _resolve_account_manager(self) -> Any:
        kclass = self._jvm["JvmClassMappingKt"].getKotlinClass(
            self._jvm["AccountManager"].class_
        )
        return self._jvm["KoinKt"].get(kclass, None, None)

    def _construct_player(self, config: HeadlessPlayerConfig) -> Any:
        offers = self._jvm["ExchangeOfferArray"](6)
        for index in range(6):
            offers[index] = self._jvm["ExchangeOffer"]()

        player_ctor = self._jvm["PlayerCtor"]
        return player_ctor.newInstance(
            [
                self._jvm["JInt"](-1),
                self._tile_id(
                    TileCoordinates(
                        x=config.tile_x,
                        y=config.tile_y,
                        level=config.tile_level,
                    )
                ),
                self._jvm["Inventories"](),
                self._jvm["HashMap"](),
                self._jvm["Experience"](),
                self._jvm["Levels"](),
                self._jvm["HashMap"](),
                self._jvm["ArrayList"](),
                None,
                None,
                config.account_name,
                "",
                self._jvm["BodyParts"](),
                offers,
                self._jvm["ArrayList"](),
                self._jvm["JInt"](0),
                None,
            ]
        )

    def _build_jvm_action(self, action: NormalizedAction) -> Any:
        if action.action_id == 0:
            return self._jvm["HeadlessWait"].INSTANCE
        if action.action_id == 1:
            if action.tile is None:
                raise BridgeError("walk_to_tile requires tile coordinates.")
            return self._jvm["HeadlessWalkToTileCtor"].newInstance(self._tile_id(action.tile))
        if action.action_id == 2:
            return self._jvm["HeadlessAttackVisibleNpc"](int(action.visible_npc_index))
        if action.action_id == 3:
            prayer = self._jvm["ProtectionPrayerById"][str(action.prayer)]
            return self._jvm["HeadlessToggleProtectionPrayer"](prayer)
        if action.action_id == 4:
            return self._jvm["HeadlessEatShark"].INSTANCE
        if action.action_id == 5:
            return self._jvm["HeadlessDrinkPrayerPotion"].INSTANCE
        if action.action_id == 6:
            return self._jvm["HeadlessToggleRun"].INSTANCE
        raise BridgeError(f"Unsupported action id: {action.action_id}")

    def _tile_id(self, tile: TileCoordinates) -> Any:
        return self._jvm["JInt"](self._jvm["TileCtorImpl"](tile.x, tile.y, tile.level))

    def _tile_from_id(self, tile_id: int) -> dict[str, int]:
        return {
            "x": int(self._jvm["TileGetX"](tile_id)),
            "y": int(self._jvm["TileGetY"](tile_id)),
            "level": int(self._jvm["TileGetLevel"](tile_id)),
        }

    def _ensure_runtime_open(self) -> None:
        if self._closed:
            raise BridgeError("HeadlessDebugClient is already closed.")
        if self._runtime is None:
            self.bootstrap_runtime()


def _ensure_jvm(headless_jar: Path, launch_cwd: Path) -> dict[str, Any]:
    global _JVM_CONTEXT
    if _JVM_CONTEXT is not None:
        expected = headless_jar.resolve()
        if _JVM_CONTEXT.classpath != expected:
            raise BridgeJVMStateError(
                "Embedded JVM is already pinned to a different classpath: "
                f"{_JVM_CONTEXT.classpath} != {expected}"
            )
        if _JVM_CONTEXT.user_dir != launch_cwd.resolve():
            raise BridgeJVMStateError(
                "Embedded JVM is already pinned to a different user.dir: "
                f"{_JVM_CONTEXT.user_dir} != {launch_cwd.resolve()}"
            )
        return _JVM_CONTEXT.classes

    try:
        import jpype
        from jpype.types import JInt
    except ModuleNotFoundError as exc:
        raise BridgeJVMStateError(
            "jpype1 is not installed in the RL environment."
        ) from exc

    if not jpype.isJVMStarted():
        jvm_library_path = resolve_jvm_library_path()
        start_jvm_kwargs: dict[str, object] = {
            "classpath": [str(headless_jar.resolve())],
        }
        if jvm_library_path is not None:
            start_jvm_kwargs["jvmpath"] = str(jvm_library_path)
        try:
            jpype.startJVM(
                f"-Duser.dir={launch_cwd.resolve()}",
                f"-Dlogback.configurationFile={QUIET_LOGBACK_CONFIG.resolve()}",
                **start_jvm_kwargs,
            )
        except jpype.JVMNotFoundException as exc:
            raise BridgeJVMStateError(
                "Could not resolve a Linux JVM runtime for the embedded bridge. "
                "Set FC_RL_JAVA_HOME or JAVA_HOME to a JDK/JRE home containing "
                "bin/java and lib/server/libjvm.so."
            ) from exc

    classes = {
        "JInt": JInt,
        "ArrayList": jpype.JClass("java.util.ArrayList"),
        "BodyParts": jpype.JClass(
            "world.gregs.voidps.engine.entity.character.player.equip.BodyParts"
        ),
        "ExchangeOffer": jpype.JClass(
            "world.gregs.voidps.engine.data.exchange.ExchangeOffer"
        ),
        "ExchangeOfferArray": jpype.JArray(
            jpype.JClass("world.gregs.voidps.engine.data.exchange.ExchangeOffer")
        ),
        "Experience": jpype.JClass(
            "world.gregs.voidps.engine.entity.character.player.skill.exp.Experience"
        ),
        "FightCaveEpisodeConfig": jpype.JClass("FightCaveEpisodeConfig"),
        "HashMap": jpype.JClass("java.util.HashMap"),
        "HeadlessAttackVisibleNpc": jpype.JClass("HeadlessAction$AttackVisibleNpc"),
        "HeadlessBatchSteppingKt": jpype.JClass("HeadlessBatchSteppingKt"),
        "HeadlessDrinkPrayerPotion": jpype.JClass("HeadlessAction$DrinkPrayerPotion"),
        "HeadlessEatShark": jpype.JClass("HeadlessAction$EatShark"),
        "HeadlessMain": jpype.JClass("HeadlessMain"),
        "HeadlessReplayStep": jpype.JClass("HeadlessReplayStep"),
        "HeadlessToggleProtectionPrayer": jpype.JClass(
            "HeadlessAction$ToggleProtectionPrayer"
        ),
        "HeadlessToggleRun": jpype.JClass("HeadlessAction$ToggleRun"),
        "HeadlessWait": jpype.JClass("HeadlessAction$Wait"),
        "HeadlessWalkToTile": jpype.JClass("HeadlessAction$WalkToTile"),
        "Inventories": jpype.JClass("world.gregs.voidps.engine.inv.Inventories"),
        "JvmClassMappingKt": jpype.JClass("kotlin.jvm.JvmClassMappingKt"),
        "KoinKt": jpype.JClass("world.gregs.voidps.engine.KoinKt"),
        "Levels": jpype.JClass(
            "world.gregs.voidps.engine.entity.character.player.skill.level.Levels"
        ),
        "AccountManager": jpype.JClass("world.gregs.voidps.engine.data.AccountManager"),
        "Player": jpype.JClass("world.gregs.voidps.engine.entity.character.player.Player"),
        "ProtectionPrayer": jpype.JClass("HeadlessProtectionPrayer"),
        "Tile": jpype.JClass("world.gregs.voidps.type.Tile"),
    }
    classes["TileCtorImpl"] = getattr(classes["Tile"], "constructor-impl")
    classes["TileGetX"] = getattr(classes["Tile"], "getX-impl")
    classes["TileGetY"] = getattr(classes["Tile"], "getY-impl")
    classes["TileGetLevel"] = getattr(classes["Tile"], "getLevel-impl")
    classes["PlayerCtor"] = _select_player_constructor(classes["Player"])
    classes["HeadlessWalkToTileCtor"] = _select_private_inline_constructor(
        classes["HeadlessWalkToTile"]
    )
    classes["ProtectionPrayerById"] = {
        str(classes["ProtectionPrayer"].ProtectFromMagic.getPrayerId()): classes["ProtectionPrayer"].ProtectFromMagic,
        str(classes["ProtectionPrayer"].ProtectFromMissiles.getPrayerId()): classes["ProtectionPrayer"].ProtectFromMissiles,
        str(classes["ProtectionPrayer"].ProtectFromMelee.getPrayerId()): classes["ProtectionPrayer"].ProtectFromMelee,
    }

    _JVM_CONTEXT = _JVMContext(
        jpype=jpype,
        classpath=headless_jar.resolve(),
        user_dir=launch_cwd.resolve(),
        classes=classes,
    )
    return classes


def _select_player_constructor(player_class: Any) -> Any:
    constructors = list(player_class.class_.getConstructors())
    for constructor in constructors:
        if len(constructor.getParameterTypes()) == 17:
            return constructor
    raise BridgeJVMStateError("Unable to locate the Player synthetic constructor.")


def _select_private_inline_constructor(action_class: Any) -> Any:
    constructors = list(action_class.class_.getDeclaredConstructors())
    for constructor in constructors:
        parameter_types = tuple(str(parameter.getName()) for parameter in constructor.getParameterTypes())
        if parameter_types == ("int",):
            constructor.setAccessible(True)
            return constructor
    raise BridgeJVMStateError(
        f"Unable to locate the private inline-value constructor for {action_class}."
    )


def _pythonize(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "getClass"):
        class_name = str(value.getClass().getName())
        if class_name == "java.lang.Boolean":
            return bool(value.booleanValue())
        if class_name in {"java.lang.Byte", "java.lang.Short", "java.lang.Integer"}:
            return int(value.intValue())
        if class_name == "java.lang.Long":
            return int(value.longValue())
        if class_name in {"java.lang.Float", "java.lang.Double"}:
            return float(value.doubleValue())
        if class_name == "java.lang.String":
            return str(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "entrySet"):
        result: dict[str, Any] = {}
        for entry in value.entrySet():
            result[str(entry.getKey())] = _pythonize(entry.getValue())
        return result
    if hasattr(value, "iterator") and not hasattr(value, "name"):
        return [_pythonize(item) for item in value]
    if hasattr(value, "name") and callable(value.name):
        return str(value.name())
    return value


def pythonize_action_result(result: Any) -> dict[str, Any]:
    return {
        "action_type": str(result.getActionType().name()),
        "action_id": int(result.getActionId()),
        "action_applied": bool(result.getActionApplied()),
        "rejection_reason": None
        if result.getRejectionReason() is None
        else str(result.getRejectionReason().name()),
        "metadata": _pythonize(result.getMetadata()),
    }


def pythonize_observation(observation: Any) -> dict[str, Any]:
    mapped = _pythonize(observation.toOrderedMap())
    validate_observation_contract(mapped)
    return mapped


def flat_observation_to_numpy(observation: Any) -> np.ndarray:
    values = np.asarray(observation.getValues(), dtype=np.float32)
    return coerce_flat_observation_row(values)
