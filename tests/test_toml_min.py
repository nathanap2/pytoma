import pathlib, importlib, textwrap

core = importlib.import_module("pytoma.core")

def _write(p, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s).lstrip("\n"), encoding="utf-8")

def test_hide_poetry_table(tmp_path: pathlib.Path):
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match=f"{(tmp_path/'proj'/'pyproject.toml').as_posix()}:tool.poetry", mode="hide")
        ],
    )
    toml = tmp_path / "proj" / "pyproject.toml"
    _write(toml, """
        [tool.poetry]
        name = "demo"
        version = "0.1.0"

        [tool.black]
        line-length = 100
    """)
    pack = core.build_prompt([tmp_path / "proj"], cfg)
    # Le bloc poetry est remplacé par un marker commenté TOML
    assert "table [tool.poetry] omitted" in pack
    # L'autre table reste visible
    assert "[tool.black]" in pack

