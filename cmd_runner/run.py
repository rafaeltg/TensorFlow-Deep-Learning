import argparse
import json
import os.path
import cmd_runner.operations as op

parser = argparse.ArgumentParser()

# Operations
parser.add_argument('-f', '--fit', dest='op', default=None, const=op.fit, help='Fit operation', action='store_const')
parser.add_argument('-p', '--predict', dest='op', default=None, const=op.predict, help='Predict operation',
                    action='store_const')
parser.add_argument('-t', '--transform', dest='op', default=None, const=op.transform, help='Transform operation',
                    action='store_const')
parser.add_argument('-r', '--reconstruct', dest='op', default=None, const=op.reconstruct, help='Reconstruct operation',
                    action='store_const')
parser.add_argument('-s', '--score', dest='op', default=None, const=op.score, help='Score operation',
                    action='store_const')
parser.add_argument('-v', '--validate', dest='op', default=None, const=op.validate, help='Validate operation',
                    action='store_const')
parser.add_argument('-o', '--optimize', dest='op', default=None, const=op.optimize, help='Optimize operation',
                    action='store_const')

# Configuration
parser.add_argument('-c', '--config', dest='config', default='', help='JSON file with the parameters of the operation')

# Output of the operation
parser.add_argument('--output', dest='output', default='',
                    help='Path to the folder where the output of the operation will be saved')

args = parser.parse_args()


def run():

    assert args.op is not None, 'Need to define some operation'
    assert args.config != '', 'Need to define the configuration file'

    configs = get_config(args.config)
    args.op(configs)


def get_config(file):
    assert os.path.isfile(file), 'Config file does not exists'
    with open(file) as data_file:
        data = json.load(data_file)

    return data


if __name__ == "__main__":
    run()
