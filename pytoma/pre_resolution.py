from .config import Config
from typing import Optional, List, Dict, Tuple


def _has_wildcards(p: str) -> bool:
    return any(ch in p for ch in "*?[")


def _specificity_score(p: str) -> int:
    # heuristic: long and with few wildcards = more specific
    return len(p) - 10 * sum(p.count(ch) for ch in "*?[")


def pre_resolve_path_rules(cfg: Config) -> tuple[Config, list[str]]:
    """
    Pre-resolves obvious conflicts between path rules before file discovery.
    - Moves literal rules above globs that cover them.
    - Makes 'hide' come before 'file:*', and 'file:*' before non-file actions on the same path.
    Returns (new_config, warnings).
    """
    rules = list(cfg.rules or [])
    if not rules:
        return cfg, []

    # Step 1: enrich with metadata
    meta = []
    for idx, r in enumerate(rules):
        is_qual = ":" in r.match
        is_path = not is_qual
        is_lit = is_path and not _has_wildcards(r.match)
        kind = r.mode
        prio_action = 3 if kind == "hide" else 2 if kind.startswith("file:") else 1
        meta.append(
            {
                "idx": idx,
                "rule": r,
                "is_path": is_path,
                "is_lit": is_lit,
                "prio_action": prio_action,
                "spec": _specificity_score(r.match) if is_path else -10_000,
            }
        )

    # Step 2: literal vs glob collisions
    warnings = []
    lit_items = [m for m in meta if m["is_path"] and m["is_lit"]]
    glob_items = [m for m in meta if m["is_path"] and not m["is_lit"]]

    # for each literal L, if a glob G matches L, mark G as "must come after L"
    must_after = set()  # pairs (idx_glob, idx_lit) => glob after literal
    for L in lit_items:
        Ls = L["rule"].match
        for G in glob_items:
            Gs = G["rule"].match
            # potential collision: L belongs to G
            import fnmatch

            if fnmatch.fnmatchcase(Ls, Gs):
                must_after.add((G["idx"], L["idx"]))
                # useful message if actions are incompatible
                if (
                    L["prio_action"] >= 2
                    and G["prio_action"] >= 2
                    and L["rule"].mode != G["rule"].mode
                ):
                    warnings.append(
                        f"Path conflict: literal '{Ls}' ({L['rule'].mode}) overlaps glob '{Gs}' ({G['rule'].mode}); "
                        "ordering adjusted: literal wins."
                    )

    # Step 3: stable sort with composite key
    # overall order: (qual vs path), then (literal first), then (action priority), then (specificity), then index
    def sort_key(m):
        return (
            0 if m["is_path"] else 1,  # path rules first
            0 if m["is_lit"] else 1,  # literals before globs
            -m["prio_action"],  # hide(3) > file:*(2) > others(1)
            -m["spec"],  # more specific first
            m["idx"],  # stable
        )

    meta_sorted = sorted(meta, key=sort_key)

    # Step 4: apply must_after constraints (small stable topological sort)
    # re-pass to push affected globs after their corresponding literal
    order = [m["idx"] for m in meta_sorted]
    pos = {idx: i for i, idx in enumerate(order)}
    changed = True
    while changed:
        changed = False
        for g, l in list(must_after):
            if pos[g] < pos[l]:
                # minimal swap: move 'l' up or 'g' down
                order.remove(g)
                order.insert(pos[l] + 1, g)
                pos = {idx: i for i, idx in enumerate(order)}
                changed = True

    new_rules = [rules[i] for i in order]
    new_cfg = Config(default=cfg.default, rules=new_rules, excludes=cfg.excludes)
    return new_cfg, warnings
