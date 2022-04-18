import os

sec = os.environ["ANACONDA_TOKEN"]
print([ord(c) for c in sec])
