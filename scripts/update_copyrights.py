import os
import subprocess


def update_copyrights(root_dir, year):
    for folder, _, files in os.walk(root_dir):
        for filename in files:
            if filename[0] != "." and os.path.splitext(filename)[1] in [
                ".py",
                ".h",
                ".c",
                ".cpp",
            ]:
                filePath = os.path.abspath(os.path.join(folder, filename))
                args = [
                    "annotate",
                    "--copyright=Intel Corporation",
                    "--license=Apache-2.0",
                    "--year",
                    str(year),
                    "--merge-copyrights",
                    filePath,
                ]
                subprocess.check_call(
                    ["reuse", *args],
                    shell=False,
                )


path = os.path.dirname(os.path.realpath(__file__))
source_path = os.path.dirname(path)

if __name__ == "__main__":
    print("Provide new copyright year:")
    year = input()
    update_copyrights(source_path + "/numba_dpex", year)
