#!/usr/bin/env python
import argparse
import sys
print("this is script.py")
K = 2
import test_file
def parse_args():
    parser=argparse.ArgumentParser(description="a script to do stuff")
    parser.add_argument("R1_file")
    args=parser.parse_args()
    return args

print("this is the main function")
