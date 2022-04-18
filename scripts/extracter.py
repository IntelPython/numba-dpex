with open("secret.txt", "r+") as sf:
    # Reading form a file
    line = sf.readline()
    print("printing line:", line)
    for c in line:
        print(c)
        print(ord(c) + 13)
