"""Check instruments in the MBO data.

Usage:
    python test_instruments.py <path-to-mbo-file>
    # or set HFT_MBO_FILE
"""
import os
import sys
import databento as db

path = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("HFT_MBO_FILE", ""))
if not path:
    raise SystemExit(
        "Provide an MBO file path as argv[1] or set HFT_MBO_FILE."
    )

store = db.DBNStore.from_file(path)
df = store.to_df()

main = df[df["instrument_id"] == 14160]
print(f"Main instrument (14160): {len(main):,} msgs")
if len(main) > 0:
    sym = main["symbol"].iloc[0]
    print(f"Symbol: {sym}")
    px = main["price"].dropna()
    print(f"Price range: {px.min():.2f} - {px.max():.2f}")
    print(f"Actions: {main['action'].value_counts().to_dict()}")
    print()

    ts = main["ts_event"]
    print(f"ts_event dtype: {ts.dtype}")
    print(f"ts_event range: {ts.min()} to {ts.max()}")
    ts_int = ts.values.astype("int64")
    print(f"ts_event as int64: {ts_int[1]} (sample)")
    print()

for iid, grp in df.groupby("instrument_id"):
    s = grp["symbol"].iloc[0]
    p = grp["price"].dropna()
    lo = f"{p.min():.1f}" if len(p) > 0 else "N/A"
    hi = f"{p.max():.1f}" if len(p) > 0 else "N/A"
    print(f"  {iid:>10d}  {s:20s}  {len(grp):>10,} msgs  price: {lo}-{hi}")
