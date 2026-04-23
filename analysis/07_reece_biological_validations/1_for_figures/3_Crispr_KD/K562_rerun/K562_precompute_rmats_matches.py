import sys, gzip, pickle
from pathlib import Path
import polars as pl
import polars.selectors as cs

# --- copy your helpers from K562_Crispr_KD_2.py ---
from K562_Crispr_KD import get_path, load_rmats

RBPS = ["NIPBL","SAFB","NOLC1","ZC3H11A","EXOSC5","FXR2","RPS3","ZNF800",
    "SDAD1","SRSF7","IGF2BP1","DDX42","MORC2","RYBP","DDX21","APEX1",
    "RPS6","DDX6","GNL3","ELAC2","NPM1","TRA2A","METTL1","PRPF8",
    "ELAVL1","XRCC6","SRSF9","ADAT1","DDX43","RPS11","EIF4E","EXOSC10",
    "RNF187","SF3B1","GARS","YWHAG", "DGCR8"
]

OUT_DIR = Path("precomputed_rmats_K562")
OUT_DIR.mkdir(exist_ok=True)

# Load BAT ONCE
print("Loading BAT...", flush=True)
BAT_K562 = pl.read_ipc(get_path("K562")).rename({"FDR": "BAT FDR"})

for rbp in RBPS:
    out_path = OUT_DIR / f"{rbp}_K562.feather"
    if out_path.exists():
        print(f"Skipping {rbp}, already exists.")
        continue
    try:
        df = load_rmats(rbp, BAT_K562, "K562", read_counts_threshold=10)
        df.write_ipc(str(out_path))
        print(f"Saved {rbp} → {out_path}  shape={df.shape}", flush=True)
    except Exception as e:
        print(f"ERROR for {rbp}: {e}", flush=True)

print("Done.")