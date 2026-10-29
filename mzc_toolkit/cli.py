"""python -m mzc_toolkit triage NET.npz [--k 16]       Stage 0, weights only
   python -m mzc_toolkit decompose NET.npz INIT.npz    Stage 1, needs the init (INIT.npz keys w<l> or init_w<l>)"""
import argparse, json, sys
from .io import load_layers
from .spectral import triage
from .decompose import decompose


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mzc_toolkit"); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage"); t.add_argument("net"); t.add_argument("--k", type=int, default=16); t.add_argument("--prefix", default="w")
    d = sub.add_parser("decompose"); d.add_argument("net"); d.add_argument("init"); d.add_argument("--init-prefix", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "triage": print(json.dumps(triage(load_layers(a.net, a.prefix), a.k), indent=1))
    else:
        import numpy as np
        z = np.load(a.init).files; pre = a.init_prefix or ("init_w" if "init_w0" in z else "w")
        print(json.dumps(decompose(load_layers(a.net), load_layers(a.init, pre)), indent=1))


if __name__ == "__main__":
    main()
