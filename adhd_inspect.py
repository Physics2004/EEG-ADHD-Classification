import scipy.io as sio
import numpy as np

FILE_PATH = "../../Data/raw/dataset1/ADHD_EEG/EEG/FADHD.mat"

# Load .mat file
data = sio.loadmat(FILE_PATH)

print("\nMAT KEYS:")
print(data.keys())

# Extract main key (ignore metadata keys)
main_keys = [k for k in data.keys() if not k.startswith("__")]
key = main_keys[0]

print("\nMain key:", key)

# Access main data
obj = data[key]

print("Structure shape:", obj.shape)
print("Total elements:", obj.size)

print("\nInspecting each element:\n")

all_lengths = []

# Loop through all elements safely
for i, element in enumerate(obj.flat):

    print(f"Element {i}")

    if isinstance(element, np.ndarray):

        print("Type:", type(element))
        print("Shape:", element.shape)

        # Try to interpret dimensions
        if len(element.shape) == 3:
            trials, time_points, channels = element.shape
            print(f"Interpretation: trials={trials}, time_points={time_points}, channels={channels}")
            all_lengths.append(time_points)

        if element.size > 0:
            print("Example data:", element.flatten()[:5])

    else:
        print("Type:", type(element))

    print()

# Summary statistics
print("====== SUMMARY ======")
if len(all_lengths) > 0:
    print("Min time length:", min(all_lengths))
    print("Max time length:", max(all_lengths))
    print("Unique lengths:", sorted(set(all_lengths)))
