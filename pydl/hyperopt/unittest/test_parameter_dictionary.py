import unittest

from pydl.hyperopt.parameter_dictionary import IntegerParameter, RealParameter, ListParameter, ParameterDictionary


class ParameterDictionaryTestCase(unittest.TestCase):

    def test_add_parameter(self):
        d = ParameterDictionary()

        self.assertRaises(AssertionError, d.add, 1)
        self.assertRaises(TypeError, d.add, {'test': 1})
        self.assertRaises(AssertionError, d.add, {'test': [1]})

        p = {'test': RealParameter(0, 1)}
        d.add(p)
        self.assertEqual(d.size, 1)

        p_list = {'test_list': [RealParameter(0, 1)]}
        d.add(p_list)
        self.assertEqual(d.size, 2)

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

    def test_get_param(self):
        self.assertRaises(AssertionError, ParameterDictionary.get_param, 1)
        self.assertRaises(AssertionError, ParameterDictionary.get_param, {"min_value": 0})
        self.assertRaises(SyntaxError, ParameterDictionary.get_param, {"type": "a"})

        self.assertEqual(ParameterDictionary.get_param({"type": "int", "min_value": 10, "max_value": 100}),
                         IntegerParameter(min_value=10, max_value=100))

        self.assertEqual(ParameterDictionary.get_param({"type": "real", "min_value": 0.1, "max_value": 0.9}),
                         RealParameter(min_value=0.1, max_value=0.9))

        self.assertEqual(ParameterDictionary.get_param({"type": "list", "values": [1, 2, 3]}),
                         ListParameter(values=[1, 2, 3]))

    def test_from_json(self):
        json = {
            "a": {"type": "int", "min_value": 0, "max_value": 10},
            "b": [
                {"type": "real", "min_value": 0.1, "max_value": 0.5}
            ]
        }

        d = ParameterDictionary()
        d.from_json(json)

        expected_dict = ParameterDictionary()
        expected_dict.add({
            "a": IntegerParameter(min_value=0, max_value=10),
            "b": [RealParameter(min_value=0.1, max_value=0.5)]
        })

        self.assertEqual(expected_dict.__dict__, d.__dict__)


if __name__ == '__main__':
    unittest.main()
