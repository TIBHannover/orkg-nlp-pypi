import glob
import os
import unittest

current_dir = os.path.dirname(os.path.realpath(__file__))

black_dirs = [
    'annotation',
    'clustering'
]


def is_under_black_dir(file):
    for black_dir in black_dirs:
        if '/{}/'.format(black_dir) in file:
            return True

    return False


def discover(root):
    all_files = glob.glob('./**/test*.py'.format(root), recursive=True)
    files = [file for file in all_files if not is_under_black_dir(file)]
    files = [file[2:-3].replace('/', '.') for file in files]

    return files


def test():
    files = discover(current_dir)
    suite = unittest.TestSuite()

    for file in files:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(file))

    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    test()
