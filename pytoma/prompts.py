from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# We keep templates short, actionable, and generic. They accept variables via {name}.
# Special placeholder: {PACK} will be replaced by the rendered repository snapshot.


@dataclass(frozen=True)
class PromptSpec:
    name: str
    description: str
    template: str  # Python .format(...) string using {PACK} and optional {vars}


# Small helper to allow optional variables without KeyError.
class _SafeVars(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _h(title: str) -> str:
    # Minimal section header to visually delimit tasks and context.
    return f"## {title}\n\n"


# --- Prompt library -----------------------------------------------------------

_PROMPTS: Dict[str, PromptSpec] = {}


def _register(p: PromptSpec) -> None:
    _PROMPTS[p.name] = p


# 1)
template_add_docstrings = "".join(
    [
        "{intro}",
        "**Focus (optionnel):** {focus_note}\n\n",
        "You are a senior Python developer. Your task is to add or improve docstrings for "
        "public functions and classes in the following packed snapshot. Keep signatures and behavior unchanged.\n\n",
        "Guidelines:\n",
        "- Prefer {docstyle} style (fallback: Google).\n",
        "- Document Args/Returns/Raises when relevant; keep examples minimal.\n",
        "- Avoid restating names; be precise and concise.\n\n",
        _h("Repository snapshot"),
        "{PACK}\n",
        _h("Your output"),
        "Return only the updated definitions with their new docstrings in fenced Python blocks. "
        "Do not include unrelated commentary unless strictly necessary.\n",
    ]
)

_register(
    PromptSpec(
        name="add_docstrings",
        description="Write high-quality docstrings for public functions/classes without changing behavior.",
        template=template_add_docstrings,
    )
)


# 2) factorisation
template_factor = "".join(
    [
        "{intro}",
        "**Focus :** {focus_note}\n\n",
        "Read the packed code snapshot and identify opportunities to factor duplicated logic, "
        "extract helpers, and reduce complexity while preserving behavior.\n\n",
        "Constraints:\n",
        "- Keep changes minimal ({change_budget}).\n",
        "- Do not alter public APIs unless a clear improvement is justified.\n",
        "- Prefer pure functions and local utilities when possible.\n\n",
        _h("Repository snapshot"),
        "{PACK}\n",
        _h("Your output"),
        "List concrete refactor suggestions with short rationale. When beneficial, show proposed replacements "
        "as minimal patches or fenced Python blocks.\n",
    ]
)

_register(
    PromptSpec(
        name="look_at_possible_factorisations",
        description="Identify opportunities to factor code and propose small refactors (French spelling alias).",
        template=template_factor,
    )
)
_PROMPTS["look_at_possible_factorizations"] = _PROMPTS[
    "look_at_possible_factorisations"
]


# 3) focus_to_config
template_focus_cfg = "".join(
    [
        "## Task\n\n",
        "You are given a **skeleton** of a Python repository (headers only; bodies omitted). "
        " Please output a config (see requirements below) which represents the files you would like"
        " to see developed in order to better perform the next tasks, "
        "knowing that we will have to focus on this (description might be in another language): {focus_note}\n\n",
        "### YAML output requirements\n",
        "- Use this schema exactly:\n",
        "  - `default: {default_mode}`\n",
        "  - `rules:` list with items of the form `{match, mode}`\n",
        '- Include one rule that expands the chosen target with `mode: "full"` (qualname preferred; wildcard ok).\n',
        '- Include a first rule `match: "**/*.py" / mode: "file:tidy"` **unless** `{tidy}` is `no`.\n',
        '- Optionally add up to `{support_k}` helper rules with `mode: "full"` if strictly necessary.\n',
        "- Do **not** write `excludes`. Output only the YAML in a fenced `yaml` code block.\n\n",
        "## Repository snapshot (skeleton)\n\n",
        "{PACK}\n",
    ]
)

_register(
    PromptSpec(
        name="focus_to_config",
        description="From a skeleton (signatures only), pick one target to develop and output a Pytoma YAML config that expands it.",
        template=template_focus_cfg,
    )
)


# in render_prompt(...), extend the defaults dict:
defaults = {
    "docstyle": "Google",
    "change_budget": "small batches",
    "default_mode": "sig",
    "tidy": "yes",
    "support_k": "2",
}


# --- API ---------------------------------------------------------------------


def render_prompt(name: str, pack_text: str, **vars) -> str:
    key = name.strip()
    if key not in _PROMPTS:
        raise ValueError(
            f"unknown prompt '{name}'. Available: {', '.join(sorted(_PROMPTS))}"
        )
    spec = _PROMPTS[key]

    defaults = {
        "intro": _h("Task"),
        "focus_note": "",
        "docstyle": "Google",
        "change_budget": "small batches",
        "default_mode": "sig",
        "tidy": "yes",
        "support_k": "2",
    }
    if vars:
        defaults.update(vars)

    defaults["PACK"] = pack_text
    safe = _SafeVars(defaults)

    return spec.template.format_map(safe)
