import os

current_dir = os.getcwd()
directory = os.listdir(current_dir + "/numba_dppy/examples/debug/commands")
os.chdir(current_dir + "/numba_dppy/examples/debug/commands")
for file in directory:
    if not "_docs" in file:
        open_file = open(file, "r")
        read_lines = open_file.readlines()
        docs = file + "_docs"
        if docs:
            os.remove(docs)
        write_file = open(docs, "a")
        for line in read_lines:
            if line.startswith("# Expected"):
                continue
            if line.startswith("# Run: "):
                line = line.replace("# Run:", "$")
            elif line.startswith("# "):
                line = line.replace("# ", "")
            else:
                line = "(gdb) " + line

            write_file.write(line)
