import numpy as np

def read_and_analyze(file_path):
    line_sums = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            parts = [float(x) for x in line.replace(',', ' ').split()]
            line_sum = sum(parts)
            line_sums.append(line_sum)

    if not line_sums:
        print("No valid data")
        return

    data = np.array(line_sums)

    print(f"processed {len(data)} lines")
    print(f"average: {np.mean(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"min: {np.min(data):.4f}")
    print(f"std: {np.std(data):.4f}")

# 使用方法
if __name__ == "__main__":
    file_path = input("enter txt file path： ").strip()
    read_and_analyze(file_path)

