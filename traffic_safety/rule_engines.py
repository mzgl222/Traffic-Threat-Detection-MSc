"""Selection of the rule engine used by run.py.

Two engines are available:
- "legacy": SafetyRuleEngine (SAFE/WARNING/DANGER), the original behaviour;
- "binary": BinaryRuleEngine (OK/DANGER, rules R1 and R2).

The default is "legacy", so existing runs behave exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace

from traffic_safety.binary_rules import BinaryRuleEngine
from traffic_safety.safety_rules import SafetyRuleEngine


LEGACY = "legacy"
BINARY = "binary"
ENGINE_NAMES = (LEGACY, BINARY)
DEFAULT_ENGINE = LEGACY


def resolve_engine_name(cli_value: str | None, config: dict) -> str:
    """CLI value > config["rules"]["engine"] > default ("legacy")."""
    if cli_value is not None:
        name = cli_value
    else:
        name = (config.get("rules") or {}).get("engine", DEFAULT_ENGINE)

    name = str(name).strip().lower()

    if name not in ENGINE_NAMES:
        raise ValueError(
            f"Unknown rule engine {name!r}; expected one of {ENGINE_NAMES}"
        )

    return name


def create_rule_engine(name: str, config: dict):
    """Build the selected engine with thresholds taken from the config.

    legacy: config["risk"] (required keys, as before).
    binary: optional config["binary_rules"]; missing keys fall back to the
            BinaryRuleEngine defaults (arrival_time_s=4.0, moving_speed_kmh=2.0).
    """
    if name == LEGACY:
        risk = config["risk"]
        return SafetyRuleEngine(
            warning_ttc_s=risk["warning_ttc_s"],
            danger_ttc_s=risk["danger_ttc_s"],
            fallback_distance_m=risk["fallback_distance_m"],
        )

    if name == BINARY:
        params = config.get("binary_rules") or {}
        allowed = {"arrival_time_s", "moving_speed_kmh"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"Unknown keys in 'binary_rules': {sorted(unknown)}"
            )
        return BinaryRuleEngine(**params)

    raise ValueError(
        f"Unknown rule engine {name!r}; expected one of {ENGINE_NAMES}"
    )


def binary_events_as_rule_results(events) -> list:
    """Adapt BinaryRiskEvent objects to the interface InteractionCsvWriter
    expects (pedestrian_id, vehicle_id, level.value, rule), so that in binary
    mode the interaction CSV columns rule_risk_level / rule_trigger describe
    the binary decision ("danger" + rule name, otherwise "ok").
    """
    return [
        SimpleNamespace(
            pedestrian_id=event.pedestrian_id,
            vehicle_id=event.vehicle_id,
            level=SimpleNamespace(value="danger"),
            rule=event.rule.value,
        )
        for event in events
    ]
