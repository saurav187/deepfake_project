"""Create a fresh baseline manifest and crosswalk without modifying historical arrays."""
import argparse
from pathlib import Path
from audio_features import RUN
from prepare_manifest import prepare
from historical_manifest import crosswalk


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=RUN)
    out=p.parse_args().out
    prepare(out)
    crosswalk(out,RUN.parents[1]/'data')


if __name__=='__main__':
    main()
