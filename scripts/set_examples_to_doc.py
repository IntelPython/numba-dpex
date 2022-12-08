""" This script is needed to convert gdb scripts from commands to documentation
"""
import os


def convert_commands_to_docs():
    commands_dir = os.getcwd() + "/numba_dpex/examples/debug/commands"
    examples = os.listdir(commands_dir)
    os.chdir(commands_dir + "/docs")
    for file in examples:
        if file != "docs":
            with open(commands_dir + "/" + file, "r") as open_file:
                read_lines = open_file.readlines()
                if os.path.exists(file):
                    os.remove(file)
                with open(file, "a") as write_file:
                    for line in read_lines:
                        if (
                            line.startswith("# Expected")
                            or line.startswith("echo Done")
                            or line.startswith("quit")
                            or line.startswith("set trace-commands")
                            or line.startswith("set pagination")
                        ):
                            continue
                        if line.startswith("# Run: "):
                            line = line.replace("# Run:", "$")
                            words = line.split()
                            for i in range(len(words)):
                                if words[i] == "-command" or words[
                                    i
                                ].startswith("commands"):
                                    words[i] = ""
                            line = " ".join(words)
                            line = " ".join(line.split()) + "\n"
                        elif line.startswith("# "):
                            line = line.replace("# ", "")
                        else:
                            line = "(gdb) " + line

                        write_file.write(line)


if __name__ == "__main__":
    convert_commands_to_docs()
