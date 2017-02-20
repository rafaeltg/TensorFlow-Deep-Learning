import unittest

from pydl.hyperopt.parameter import *


class HyperOptSpaceTestCase(unittest.TestCase):

    def test_size(self):
        space = hp_space({})
        self.assertEqual(space.size, 0)

        space = hp_space({
            'model': hp_choice([
                {
                    'class_name': 'MLP',
                    'config': {
                        'layers': [hp_int(10, 100), hp_int(10, 100)],
                        'activation': hp_choice(['relu', 'sigmoid', 'tanh'])
                    }
                },
                {
                    'class_name': 'MLP',
                    'config': {
                        'layers': [hp_int(10, 100)],
                        'activation': hp_choice(['relu', 'sigmoid', 'tanh'])
                    }
                }
            ])
        })
        self.assertEqual(space.size, 4)


if __name__ == '__main__':
    unittest.main()