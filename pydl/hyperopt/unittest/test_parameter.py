import unittest
from pydl.hyperopt.parameter import *


class ParametersTestCase(unittest.TestCase):

    def test_hp_choice(self):
        self.assertRaises(AssertionError, hp_choice, 1)
        self.assertRaises(AssertionError, hp_choice, [])

        x = hp_choice([1, 2, 3])
        self.assertEqual(x.size, 1)
        self.assertEqual(x.get_value([0.5]), 2)

        a = hp_int(1, 4)
        b = hp_int(1, 4)
        x = hp_choice([a, b])
        self.assertEqual(x.size, 2)
        self.assertEqual(x.get_value([0, 0]), 1)

        x = hp_choice([hp_choice([a, b]), a, 'lala'])
        self.assertEqual(x.size, 3)
        self.assertEqual(x.get_value([1, 1, 0]), 'lala')

    def test_hp_int(self):
        min_v = 1
        max_v = 10

        self.assertRaises(AssertionError, hp_int, max_v, min_v)

        p = hp_int(min_v, max_v)

        self.assertEqual(p.size, 1)
        self.assertEqual(min_v, p.get_value([0]))
        self.assertEqual(max_v, p.get_value([1]))
        self.assertEqual(5, p.get_value([0.5]))

    def test_hp_float(self):
        min_v = 0
        max_v = 1

        self.assertRaises(AssertionError, hp_float, max_v, min_v)

        p = hp_float(min_v, max_v)

        self.assertEqual(p.size, 1)
        self.assertEqual(min_v, p.get_value([0]))
        self.assertEqual(max_v, p.get_value([1]))
        self.assertEqual(0.5, p.get_value([0.5]))

    def test_hp_space(self):
        self.assertRaises(AssertionError, hp_space, 1)

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