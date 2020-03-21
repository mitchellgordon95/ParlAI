import fileinput

for line in fileinput.input():
    parts = line.strip("\n").split("\t")
    if len(parts) != 2:
        print(line)
    else:
        print(f"{parts[0][0]} {parts[1]}\t{parts[0][2:]}")

