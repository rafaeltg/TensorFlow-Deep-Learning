import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fit', dest='fit', default=False, help='Fit operation', action='store_true')
parser.add_argument('-p', '--predict', dest='predict', default=False, help='Predict operation', action='store_true')
parser.add_argument('-t', '--transform', dest='transform', default=False, help='Transform operation',
                    action='store_true')
parser.add_argument('-r', '--reconstruct', dest='reconstruct', default=False, help='Reconstruct operation',
                    action='store_true')
parser.add_argument('-s', '--score', dest='score', default=False, help='Score operation', action='store_true')

args = parser.parse_args()
print(args.fit)
