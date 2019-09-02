from ctypes import *

class Node(Structure): pass
Node._fields_ = [
    ("leaf", c_int),
    ("g", c_float),
    ("min_samples", c_int),
    ("split_ind", c_int),
    ("split", c_float),
    ("left", POINTER(Node)),
    ("right", POINTER(Node))]

trees = CDLL("./trees.so")

trees.get_root.argtypes = (c_int, )
trees.get_root.restype = POINTER(Node)

class Tree(object):
    def __init__(self, min_samples=1):
        self.root = trees.get_root(min_samples)

if __name__ == '__main__':
    tree = Tree()
