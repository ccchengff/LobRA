import os
import csv
import numpy as np

def write_to_csv(row, name):
    filename = name
    fieldnames = list(row.keys())
    if os.path.dirname(filename) != "" and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    # Check if file exists and is empty
    file_exists = os.path.isfile(filename) and os.stat(filename).st_size != 0
    
    with open(filename, 'a', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(row)

def read_from_csv(name):
    filename = name
    rows = []
    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
        return rows
    with open(filename, mode='r') as f:
        print(f"openning {filename}")
        reader = csv.reader(f)
        header = next(reader)
        key_list = header
        rows = []
        for row in reader:
            row_dict = {}
            for key, value in zip(key_list, row):
                row_dict[key] = eval(value)
            if np.nan in row_dict.values():
                continue
            rows.append(row_dict)
    return rows
