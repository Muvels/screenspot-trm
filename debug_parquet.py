import pyarrow.parquet as pq
import sys

path = "dataset/screenspot_tokenized.parquet"

print(f"Inspecting {path}...")
try:
    pf = pq.ParquetFile(path)
    print(f"Schema: {pf.schema_arrow}")
except Exception as e:
    print(f"Failed to open ParquetFile: {e}")
    sys.exit(1)

cols_to_test = ["task", "input_ids", "attention_mask", "bbox", "image"]

for col in cols_to_test:
    print(f"Testing reading column: {col}...")
    try:
        t = pq.read_table(path, columns=[col])
        print(f"  Success! Rows: {t.num_rows}")
        # Try accessing one
        # print(f"  First item: {t[col][0]}")
    except Exception as e:
        print(f"  FAILED: {e}")
