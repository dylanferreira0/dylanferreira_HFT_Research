"""Quick schema check on one MBO file.

Usage:
    python test_schema.py <path-to-mbo-file>
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

meta = store.metadata
print("Schema:", meta.schema)
print("Symbols:", getattr(meta, 'symbols', 'N/A'))
print("SType:", getattr(meta, 'stype_in', 'N/A'))
print()

df = store.to_df()
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Index: name={df.index.name}, dtype={df.index.dtype}")
print()

print("dtypes:")
for c in df.columns:
    print(f"  {c:25s} {df[c].dtype}")
print()

print("First 3 rows:")
print(df.head(3).to_string())
print()

print("Unique actions:", df['action'].unique()[:10])
print("Unique sides:", df['side'].unique()[:10])
print(f"Action type: {type(df['action'].iloc[0])} value: {repr(df['action'].iloc[0])}")
print(f"Side type: {type(df['side'].iloc[0])} value: {repr(df['side'].iloc[0])}")
print()

iid = df['instrument_id']
print(f"Unique instrument_ids: {iid.nunique()}")
print("Top 5 by count:")
print(iid.value_counts().head())
print()

print(f"Price range: {df['price'].min()} to {df['price'].max()}")
print(f"Price samples: {df['price'].iloc[:5].values}")
print(f"Size samples: {df['size'].iloc[:5].values}")
print(f"Order ID samples: {df['order_id'].iloc[:5].values}")
