"""Rule engine: load rules.yaml, run the knowledge base, fire checks, record evidence (report 5.3.6)."""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import yaml

from ..schema import Flag, Holding, RegisterRow
from .kb import FACT_TYPES, KB, forward_chain, load_facts


class RuleError(ValueError):
    pass


# ----------------------------------------------------------------------------- safe expressions

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Compare, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Name, ast.Load,
    ast.Attribute, ast.Constant, ast.Call, ast.IfExp, ast.Tuple, ast.List,
)
_FUNCS = {"abs": abs, "max": max, "min": min, "len": len}
_NAMES = {"fact", "params", "True", "False", "None", *_FUNCS}


def compile_check(expr: str, rule_id: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RuleError(f"{rule_id}: bad check expression {expr!r}: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleError(f"{rule_id}: '{type(node).__name__}' is not allowed in checks")
        if isinstance(node, ast.Name) and node.id not in _NAMES:
            raise RuleError(f"{rule_id}: unknown name '{node.id}' (use fact.<field> or params.<name>)")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise RuleError(f"{rule_id}: private attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise RuleError(f"{rule_id}: only {sorted(_FUNCS)} may be called")
    return compile(tree, f"<rule {rule_id}>", "eval")


# ----------------------------------------------------------------------------- rule set


@dataclass
class Rule:
    id: str
    name: str
    kind: str
    severity: str
    fact: str
    check: str
    code: Any = field(repr=False, default=None)


@dataclass
class RuleSet:
    rules: list[Rule]
    params: dict
    contradiction_severities: set[str]
    profile: str = "default"

    def get(self, rule_id: str) -> Rule:
        return next(r for r in self.rules if r.id == rule_id)


def load_rules(path: str | Path = "rules.yaml", profile: Optional[str] = None,
               overrides: Optional[dict] = None) -> RuleSet:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    params = dict(data.get("params", {}))
    if profile and profile != "default":
        profiles = data.get("profiles", {})
        if profile not in profiles:
            raise RuleError(f"unknown profile '{profile}'; available: {sorted(profiles)}")
        params.update(profiles[profile])
    params.update(overrides or {})
    rules = []
    for r in data["rules"]:
        missing = {"id", "name", "kind", "severity", "fact", "check"} - set(r)
        if missing:
            raise RuleError(f"rule {r.get('id', '?')} lacks {sorted(missing)}")
        if r["fact"] not in FACT_TYPES:
            raise RuleError(f"{r['id']}: unknown fact type '{r['fact']}'; known: {sorted(FACT_TYPES)}")
        rules.append(Rule(**{k: r[k] for k in ("id", "name", "kind", "severity", "fact", "check")},
                          code=compile_check(str(r["check"]), r["id"])))
    if len({r.id for r in rules}) != len(rules):
        raise RuleError("duplicate rule ids")
    return RuleSet(rules, params, set(data.get("contradiction_severities", ["high", "medium"])),
                   profile or "default")


# ----------------------------------------------------------------------------- evaluation


def _jsonable(v):
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator}"
    if isinstance(v, (set, frozenset)):
        return sorted(_jsonable(x) for x in v)
    return v


def evidence_of(fact) -> dict:
    return {k: _jsonable(v) for k, v in dataclasses.asdict(fact).items() if k != "allowed"}


def validate(h: Holding, row: Optional[RegisterRow], rules: RuleSet,
             khasra_index: Optional[dict] = None) -> list[Flag]:
    """Flags for one holding against its linked register row (None when UNLINKED: intra rules only)."""
    kb = KB()
    load_facts(kb, h, row, khasra_index)
    forward_chain(kb, rules.params)
    params = SimpleNamespace(**rules.params)
    flags = []
    for rule in rules.rules:
        for fact in kb.all(FACT_TYPES[rule.fact]):
            env = {"fact": fact, "params": params, **_FUNCS}
            try:
                fired = bool(eval(rule.code, {"__builtins__": {}}, env))  # noqa: S307 - validated AST
            except TypeError:
                fired = False   # e.g. comparing a missing value; such cases are covered by R1-R4
            if fired:
                flags.append(Flag(rule=rule.id, name=rule.name, kind=rule.kind, severity=rule.severity,
                                  holding_id=h.holding_id, evidence=evidence_of(fact)))
    return flags


def is_contradiction(flags: list[Flag], rules: RuleSet) -> bool:
    return any(f.severity in rules.contradiction_severities for f in flags)
