import sys, gzip, pickle
from pathlib import Path
import polars as pl
import polars.selectors as cs

# --- copy your helpers from HepG2_Crispr_KD.py ---
from HepG2_Crispr_KD import get_path, load_rmats

RBPS = ['SAFB', 'NOLC1', 'ZC3H11A', 'EXOSC5', 'WDR43',
        'RBM5', 'FXR2', 'DROSHA', 'STAU2', 
        'CDC40', 'SRSF7', 'EIF3H', 'SDAD1', 'IGF2BP1', 'AGGF1']

OUT_DIR = Path("precomputed_rmats_HepG2")
OUT_DIR.mkdir(exist_ok=True)

# Load BAT ONCE
print("Loading BAT...", flush=True)
BAT_HepG2 = pl.read_ipc(get_path("HepG2")).rename({"FDR": "BAT FDR"})

for rbp in RBPS:
    out_path = OUT_DIR / f"{rbp}_HepG2.feather"
    if out_path.exists():
        print(f"Skipping {rbp}, already exists.")
        continue
    try:
        df = load_rmats(rbp, BAT_HepG2, "HepG2", read_counts_threshold=10)
        df.write_ipc(str(out_path))
        print(f"Saved {rbp} → {out_path}  shape={df.shape}", flush=True)
    except Exception as e:
        print(f"ERROR for {rbp}: {e}", flush=True)

print("Done.")