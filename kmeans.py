import sys
import numpy

def main():
    args = sys.argv[1:]

    if not args or len(args) != 1:
        print("usage: main filename")
        sys.exit(1)
    
    filename = args[0]


if __name__ == '__main__':
  main()
