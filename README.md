# Pytoma — Repo text rendering

Pytoma condenses your python repo into one concise, LLM-friendly, text file.
You choose **what to show** (full code, signature + docstring, partial bodies ...) **per function, class, file, or folder**. 

You can configure what will be shown via a small YAML config or CLI options, and the output is a single text file divided in fenced code blocks, ideal for prompting an LLM or sharing a focused snapshot with collaborators. You can also use it to hide sensitive code snippets to protect your innovative ideas.

Note from the author : I originally vibe-coded a basic tool for myself because ChatGPT wouldn't accept more than 10 files and the files-to-prompts lib didn't allow me to hide certain sections ... but it turned out to be very handy and useful, so I refined it and it became this project.


---

## Highlights

* **Policy-driven slicing**: `full`, `sig`, `sig+doc`, `body:levels=k`, `hide` to omit sections, `file:no-imports` to omit imports
* **Targets**:

  * by **qualname** (e.g. `pkg.mod:Class.method` or `pkg.mod:*`),
  * by **path glob** (e.g. `**/repo/pkg/**/*.py`),
  * or via a **global default**.
* **Multiple engines**:

  * Python (`.py`): granular per function/method/class.
  * Markdown (`.md`): per heading or whole document (supports `full`/`hide`).
  * YAML will soon be added
  * A description or short extract of heavy files, ML artifacts, etc., is also to be expected

---

## Installation

```bash
# For now, in a freshly cloned version of pytoma : 
pip install -e .
# There will be a more official release soon
```

> Requires Python ≥ 3.9 (uses `ast.unparse`) and the following deps: `libcst`, `pyyaml`.

---

## Quick Start (CLI)

```bash
# Basic: pack a repo with defaults (same behaviour as files-to-prompt)
pytoma ./path/to/repo > PACK.txt

# Now we use a YAML config to customize the render (skip sections, etc.)
pytoma ./repo --config config.yml --out PACK.txt

```

The pack looks like:

````text
### pkg/module.py

```python
def foo(a: int, b: int) -> int:
    """Adds two numbers."""
    # … lines 12–37 body omitted (26 lines) …
```

### docs/guide.md

<!-- … lines 1–120 document omitted (120 lines) … -->

````


---

## Configuration (YAML)

```yaml
# config.yml
default: full        # fallback policy

# optional: exclude patterns (relative to the provided roots)
excludes:
  - ".venv/**"
  - "**/__pycache__/**"
  - "dist/**"
  - "build/**"
  - "site-packages/**"
  - "**/*.pyi"

rules:
  # Apply a file-level filter first (removes top-level imports, keeps __future__)
  - match: "/abs/path/to/repo/pkg/**/*.py"
    mode: "file:no-imports"

  # Then apply function-level policy to the same files
  - match: "/abs/path/to/repo/pkg/**/*.py"
    mode: "sig+doc"

  # Hide an entire module (replaced by an omission marker)
  - match: "/abs/path/to/repo/legacy/big_module.py"
    mode: "hide"

  # Target all defs in a given module by qualname
  - match: "pkg.special:*"
    mode: "body:levels=1"

  # Markdown supports only "full" and "hide"
  - match: "/abs/path/to/repo/README.md"
    mode: "hide"
````

Note : when Pytoma walks each document’s nodes, it decides a policy with this order: qualname rules > path rules > global default.

---

## Policies

* `full` — leave the node as is.
* `hide` — remove and insert an omission marker:

  * On a **module**: replaces the whole file content.
  * On a **class**: replaces the whole class block.
  * On a **function/method**: replaces the definition or its body depending on engine mode.
* `sig` — one-line header (`def …:`), body omitted with a marker.
* `sig+doc` — header + docstring (or `"""…"""` placeholder), body omitted.
* `body:levels=k` — keep code with indentation ≤ `base + 4*k`, collapse deeper blocks with markers.
* `file:no-imports` — strip top-level `import`/`from … import …` except `from __future__ import …`.

Markers are comment-style lines (or a light box) with line counts, e.g.:

```python
# … lines 42–97 body omitted (56 lines) …
```


---

## Applying a policy to a whole file or folder

**Three options:**

1. **Global default**

```bash
pytoma /abs/repo --default "sig+doc"
```

2. **Path glob rule** (absolute POSIX)

```yaml
rules:
  - match: "/abs/repo/pkg/**/*.py"
    mode: "sig+doc"
```

3. **Qualname wildcard for a module**

```yaml
rules:
  - match: "pkg.module:*"
    mode: "sig+doc"
```

> For file-level effects like removing imports or hiding an entire file, use `file:no-imports` or `hide` on the path glob; these apply to the module root.

---

## Excludes

By default, Pytoma skips:

```
.venv/**, **/__pycache__/**, dist/**, build/**, site-packages/**, **/*.pyi
```

You can override in `config.yml → excludes`.

---

## Pitfalls

- **Rule order can still be tricky** when several rules are about the same file. I'm looking forward to improving that soon. For the moment, prefer being as explicit as possible, and for tricky setups, you can simply ask an LLM to draft the rules for you (from your repo tree and goals).
- To name a markdown section in the config file, you still **need to give the slugified version** of the section

## Roadmap

* Engine for **YAML / JSON** (keys/tables granularity), **CSV** (short preview like pandas does), etc.
* Better summary of skipped lines (for instance, when skipping imports, in the examples of skipped imports, we should probably prioritize unexpected imports over classic ones such as sys, os ...)
* Propose the automatic addition of inferred rules: in particular, functions that are only called by hidden functions should probably be hidden by default.
* More markers + Let the user choose their markers, rule by rule, using the configuration
* A “custom:alternative_version” rule to simply replace a block with an alternative version of the block provided by the user.
* Heuristics to suggest rules (and possibility to condition it by a section which we want to focus on) -> may actually become another independent tool ... 
* ... and even smarter module targeting via **dependency graphs , embeddings**, git edit correlations, etc.

---

## Current engines & support matrix

| Engine / Mode    | full | hide | sig | sig+doc | body\:levels=k | file\:no-imports |
| ---------------- | :--: | :--: | :-: | :-----: | :------------: | :--------------: |
| Python (`.py`)   |   ✅  |   ✅  |  ✅  |    ✅    |        ✅       |         ✅        |
| Markdown (`.md`) |   ✅  |   ✅  |  –  |    –    |        –       |         –        |

Planned: YAML/TOML section policies (toml already started)


---

## How it works (short)

* Each engine parses a file into a lightweight IR:

  * `Document` with a flat list of `Node`s (module, class, function, etc.).
* A decision step maps nodes to `Action`s (`full`, `sig+doc`, …) using the precedence above.
* Engines render actions into concrete `Edit`s (byte spans + replacements).
* Edits are **merged**.
* The final pack concatenates per-file sections with language fences.

---

## License

MIT 

---

## FAQ

**Q: Can I hide a whole file?**
Yes: `mode: "hide"` on the file path glob. You’ll get a single omission marker for the module.

**Q: Can I strip imports but keep function bodies?**
Yes: chain two rules on the same glob—first `file:no-imports`, then `sig`/`full`/`levels`, etc.

**Q: Do rules need to be ordered?**
Prefer placing `file:*` rules **before** function/method rules on the same path glob. Qualname rules are checked first anyway.

**Q: What happens to docstrings in `body:levels=k`?**
If a docstring falls in an omitted range, it’s re-inserted above the kept code for readability.

**Q: Can I get relative paths in headers?**
Yes: set `DISPLAY_PATH_MODE = "strip"` in `core.py` (default in our dev builds). (will be parametrized soon)

