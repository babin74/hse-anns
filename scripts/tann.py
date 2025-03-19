#!/usr/bin/env python3

import argparse, sys
import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tann.py')
    parser.add_argument('-v', '--verbose', action='store_true')
    subparsers = parser.add_subparsers(required=True)

    download.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)