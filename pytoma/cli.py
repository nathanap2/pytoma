import argparse, pathlib, sys
from .core import Config, build_prompt

def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-function code slicer for LLM prompts")
    ap.add_argument("paths", nargs="+", type=pathlib.Path, help="Python files or directories")
    ap.add_argument("--config", type=pathlib.Path, default=None, help="YAML file with default/rules")
    ap.add_argument("--default", type=str, default="full", help="Default mode if no rule matches")
    ap.add_argument("--verbose", action="store_true", help="Diagnostic logs in output")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="Write output to file instead of stdout")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config, args.default)
    text = build_prompt([p.resolve() for p in args.paths], cfg, verbose=args.verbose)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

if __name__ == "__main__":
    main()

