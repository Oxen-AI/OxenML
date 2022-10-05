import os
import sys
import shutil

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <input_dir> <test_dir> <num_test>")
    exit()

input_dir = sys.argv[1]
test_dir = sys.argv[2]
num_test = int(sys.argv[3])

if not os.path.exists(input_dir):
    print(f"Directory does not exist {input_dir}")
    exit()

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

src_files = []
for filename in os.listdir(input_dir):
    src_file = os.path.join(input_dir, filename)
    # checking if it is a file
    if os.path.isfile(src_file) and "_" in filename and ".mp4" in filename:
        src_files.append(src_file)

src_files.sort()

test_counts = {}
for src_file in src_files:
    filename = os.path.basename(src_file)
    category = filename.split("_")[0]

    if not category in test_counts:
        test_counts[category] = 0

    test_count = test_counts[category]

    if test_count < num_test:
        dst_file = os.path.join(test_dir, filename)
        shutil.move(src_file, dst_file)
        test_counts[category] += 1

print(f"Test counts: {test_counts}")
print("Done.")
