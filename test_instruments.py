"""Check instruments in the MBO data."""
import databento as db

store = db.DBNStore.from_file(
    r"C:\Users\Dylan Ferreira\OneDrive\ES Datebento\glbx-mdp3-20250827.mbo.dbn.zst"
)
df = store.to_df()

main = df[df["instrument_id"] == 14160]
print(f"Main instrument (14160): {len(main):,} msgs")
sym = main["symbol"].iloc[0]
print(f"Symbol: {sym}")
px = main["price"].dropna()
print(f"Price range: {px.min():.2f} - {px.max():.2f}")
print(f"Actions: {main['action'].value_counts().to_dict()}")
print()

# ts_event check
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
