# -*- coding: utf-8 -*-
import glob
import os
import unittest
from argparse import ArgumentParser

current_dir = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--ignore",
        nargs="+",
        required=False,
        help="directory names to be ignored.",
    )

    return parser.parse_args()


def is_under_ignored_dir(file, ignored_dirs):
    for ignored_dir in ignored_dirs:
        if "/{}/".format(ignored_dir) in file:
            return True

    return False


def discover(root, ignored_dirs=None):
    files = glob.glob("./**/test*.py".format(root), recursive=True)  # noqa: F523

    if ignored_dirs:
        files = [file for file in files if not is_under_ignored_dir(file, ignored_dirs)]

    files = [file[2:-3].replace("/", ".") for file in files]

    return files


def test(args):
    files = discover(current_dir, args.ignore)
    suite = unittest.TestSuite()

    for file in files:
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(file))

    unittest.TextTestRunner(verbosity=2).run(suite)


def main():
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
