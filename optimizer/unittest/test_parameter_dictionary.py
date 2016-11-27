import unittest

from optimizer.parameter_dictionary import IntegerParameter, RealParameter, ListParameter, ParameterDictionary


class ParameterDictionaryTestCase(unittest.TestCase):

    def test_int_parameter(self):
        min_v = 1
        max_v = 10

        p = IntegerParameter(min_value=min_v, max_value=max_v)

        self.assertEqual(min_v, p.get_value(0))
        self.assertEqual(max_v, p.get_value(1))
        self.assertEqual(5, p.get_value(0.5))
        self.assertRaises(AssertionError, p.get_value, 2)

    def test_real_parameter(self):
        min_v = 0
        max_v = 1

        p = RealParameter(min_value=min_v, max_value=max_v)

        self.assertEqual(min_v, p.get_value(0))
        self.assertEqual(max_v, p.get_value(1))
        self.assertEqual(0.5, p.get_value(0.5))
        self.assertRaises(AssertionError, p.get_value, 2)

    def test_list_parameter(self):
        values = ["a", "b", "c"]

        p = ListParameter(values=values)

        self.assertEqual(values[0], p.get_value(0))
        self.assertEqual(values[2], p.get_value(1))
        self.assertEqual(values[1], p.get_value(0.5))
        self.assertRaises(AssertionError, p.get_value, 2)

    def test_add_parameter(self):
        d = ParameterDictionary()

        self.assertRaises(AssertionError, d.add, 1)
        self.assertRaises(TypeError, d.add, {'test': 1})
        self.assertRaises(AssertionError, d.add, {'test': [1]})

        p = {'test': RealParameter(0, 1)}
        d.add(p)

        p_list = {'test_list': [RealParameter(0, 1)]}
        d.add(p_list)

    def test_get_parameter(self):
        d = ParameterDictionary()
        p = {
            'test': RealParameter(0, 1),
            'test_list': [
                IntegerParameter(10, 20),
                IntegerParameter(5, 10)
            ]
        }

        d.add(p)

        self.assertRaises(AssertionError, d.get, [0])

        p_ret = d.get([0, 0, 0])
        self.assertTrue(isinstance(p_ret, dict))
        self.assertEqual(len(p_ret), 2)

        self.assertIn('test', p_ret)
        self.assertEqual(p_ret['test'], 0)

        self.assertIn('test_list', p_ret)
        self.assertTrue(isinstance(p_ret['test_list'], list))
        self.assertListEqual(p_ret['test_list'], [10, 5])

if __name__ == '__main__':
    unittest.main()
