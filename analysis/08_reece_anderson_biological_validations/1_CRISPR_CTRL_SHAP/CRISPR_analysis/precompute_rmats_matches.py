import sys, gzip, pickle
from pathlib import Path
import polars as pl
import polars.selectors as cs
from CRISPR_KD import get_path, load_rmats

# ---------------------------------------------------------------------------
# RBP lists — one per cell line
# ---------------------------------------------------------------------------
RBPS_HepG2 = ['SAFB', 'NOLC1', 'ZC3H11A', 'EXOSC5', 'WDR43',
        'RBM5', 'FXR2', 'DROSHA', 'STAU2', 
        'CDC40', 'SRSF7', 'EIF3H', 'SDAD1', 'IGF2BP1', 'AGGF1']

RBPS_K562 = ["NIPBL","SAFB","NOLC1","ZC3H11A","EXOSC5","FXR2","RPS3","ZNF800",
    "SDAD1","SRSF7","IGF2BP1","DDX42","MORC2","RYBP","DDX21","APEX1",
    "RPS6","DDX6","GNL3","ELAC2","NPM1","TRA2A","METTL1","PRPF8",
    "ELAVL1","XRCC6","SRSF9","ADAT1","DDX43","RPS11","EIF4E","EXOSC10",
    "RNF187","SF3B1","GARS","YWHAG", "DGCR8"
]

CELL_LINE_CONFIG = {
    "HepG2": RBPS_HepG2,
    "K562":  RBPS_K562,
}

# ---------------------------------------------------------------------------
# Parse cell line from command line
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python3.11 precompute_rmats_matches.py <cell_line>")
    print(f"  Valid cell lines: {list(CELL_LINE_CONFIG.keys())}")
    sys.exit(1)

cell_line = sys.argv[1]

if cell_line not in CELL_LINE_CONFIG:
    print(f"ERROR: Unknown cell line '{cell_line}'. "
          f"Valid options: {list(CELL_LINE_CONFIG.keys())}")
    sys.exit(1)

RBPS = CELL_LINE_CONFIG[cell_line]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path(f"precomputed_rmats_{cell_line}")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load BAT once for this cell line
# ---------------------------------------------------------------------------
print(f"Loading BAT for {cell_line}...", flush=True)
BAT = pl.read_ipc(get_path(cell_line)).rename({"FDR": "BAT FDR"})

# ---------------------------------------------------------------------------
# Process each RBP
# ---------------------------------------------------------------------------
for rbp in RBPS:
    out_path = OUT_DIR / f"{rbp}_{cell_line}.feather"
    if out_path.exists():
        print(f"Skipping {rbp}, already exists.")
        continue
    try:
        df = load_rmats(rbp, BAT, cell_line, read_counts_threshold=10)
        df.write_ipc(str(out_path))
        print(f"Saved {rbp} → {out_path}  shape={df.shape}", flush=True)
    except Exception as e:
        print(f"ERROR for {rbp}: {e}", flush=True)

print(f"Done with {cell_line}.")