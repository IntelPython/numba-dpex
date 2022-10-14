import os


def llvm_spirv_path():

    """return llvm-spirv path"""

    result = os.path.dirname(__file__) + "/llvm-spirv"

    return result
