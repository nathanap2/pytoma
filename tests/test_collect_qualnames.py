import importlib
import textwrap

import libcst as cst
from libcst import metadata

collect = importlib.import_module("pytoma.collect")


def _collect_funcs(code: str, modname: str = "m"):
    """
    Helper: parse `code`, run FuncCollector(modname, source_lines, posmap),
    and return the collected FuncInfo list.
    """
    src = textwrap.dedent(code)
    module = cst.parse_module(src)
    wrapper = metadata.MetadataWrapper(module)
    posmap = wrapper.resolve(metadata.PositionProvider)  # <-- new
    collector = collect.FuncCollector(modname, src.splitlines(keepends=True), posmap)
    wrapper.visit(collector)
    return collector.funcs



def test_hierarchical_qualnames_for_nested_and_methods():
    """
    The collector should emit hierarchical qualnames:
      - m:outer and m:outer.inner          (free function + nested function)
      - m:Cls.method and m:Cls.method.inner (method + nested function inside the method)
      - m:top                               (simple top-level function)
    It should NOT emit the legacy flat name m:inner.
    """
    code = """
    class Cls:
        def method(self):
            def inner():
                return 1
            return inner()

    def outer():
        def inner():
            return 2
        return inner()

    def top():
        return 3
    """
    funcs = _collect_funcs(code, modname="m")
    qnames = sorted(f.qualname for f in funcs)

    # Expected hierarchical qualnames are present
    assert "m:Cls.method" in qnames
    assert "m:Cls.method.inner" in qnames
    assert "m:outer" in qnames
    assert "m:outer.inner" in qnames
    assert "m:top" in qnames

    # Legacy flat/ambiguous qualname must not be produced anymore
    assert "m:inner" not in qnames

    # Sanity check: every qualname should be prefixed by module:
    assert all(name.startswith("m:") for name in qnames)

