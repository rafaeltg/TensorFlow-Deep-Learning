#!/usr/bin/env python3.5

import argparse
import os
from pydl.models.utils import load_json
from cli import __version__
from cli.commands import *


def parse_args():
    #parsing
    description = 'A CLI tool to help with using the pydl package'
    parser = argparse.ArgumentParser(prog='pydl', description=description)
    parser.add_argument('--version', action="version", version=__version__)

    # create global arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-c', '--config', dest='config', default='', required=True,
                        help='JSON file with the parameters of the operation')
    common.add_argument('-o', '--output', dest='output', default=os.getcwd(),
                        help='Path to the folder where the output of the operation will be saved')

    # Parser for each command
    subparsers = parser.add_subparsers(help='commands')
    subparsers.add_parser('fit', parents=[common], help='Fit operation').set_defaults(func=fit)
    subparsers.add_parser('predict', parents=[common], help='Predict operation').set_defaults(func=predict)
    subparsers.add_parser('predict_proba', parents=[common], help='Predict Probabilities operation').set_defaults(func=predict_proba)
    subparsers.add_parser('transform', parents=[common], help='Transform operation').set_defaults(func=transform)
    subparsers.add_parser('reconstruct', parents=[common], help='Reconstruct operation').set_defaults(func=reconstruct)
    subparsers.add_parser('score', parents=[common], help='Score operation').set_defaults(func=score)
    subparsers.add_parser('cv', parents=[common], help='Cross-Validation operation').set_defaults(func=cv)
    subparsers.add_parser('optimize', parents=[common], help='Optimize operation').set_defaults(func=optimize)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(load_json(args.config), args.output)


if __name__ == "__main__":
    main()
