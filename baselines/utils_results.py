import csv
from pathlib import Path

def append_result_row(csv_path, row_dict, field_order=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    if field_order is None:
        field_order = list(row_dict.keys())
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

