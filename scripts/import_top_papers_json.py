from pathlib import Path
from scireason.integrations.top_papers_import import export_meta_files

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSON file exported by top-papers-bot")
    ap.add_argument("--out", dest="out", default="configs/top_papers_meta", help="Output dir for meta files")
    args = ap.parse_args()

    out_files = export_meta_files(Path(args.inp), Path(args.out))
    print(f"Generated {len(out_files)} meta files in {args.out}")
