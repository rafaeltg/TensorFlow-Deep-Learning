import argparse

parser = argparse.ArgumentParser()

# Operations
parser.add_argument('-f', '--fit', dest='fit', default=False, help='Fit operation', action='store_true')
parser.add_argument('-p', '--predict', dest='predict', default=False, help='Predict operation', action='store_true')
parser.add_argument('-t', '--transform', dest='transform', default=False, help='Transform operation',
                    action='store_true')
parser.add_argument('-r', '--reconstruct', dest='reconstruct', default=False, help='Reconstruct operation',
                    action='store_true')
parser.add_argument('-s', '--score', dest='score', default=False, help='Score operation', action='store_true')
parser.add_argument('-v', '--validate', dest='validate', default=False, help='Validate operation', action='store_true')
parser.add_argument('-o', '--optimize', dest='optimize', default=False, help='Optimize operation', action='store_true')

# Configuration
parser.add_argument('-c', '--config', dest='config', default='', help='JSON file with the parameters of the operation')

# Output of the operation
parser.add_argument('--output', dest='output', default='',
                    help='Path to the folder where the output of the operation will be saved')

args = parser.parse_args()
print(args.fit)
