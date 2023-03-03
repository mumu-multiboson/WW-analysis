import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jobs', nargs='+')
    args = parser.parse_args()
    for j in args.jobs:
        subprocess.run(['condor_submit', str(j)])

if __name__ == '__main__':
    main()