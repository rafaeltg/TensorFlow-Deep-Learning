from .parameter import *


class HyperOptSpace(object):

    def __init__(self, space_config):
        assert isinstance(space_config, dict), 'Invalid space type. Must be <type dict>.'
        self._space = nodefy(space_config)
        self._size = calc_size(space_config)

    @property
    def size(self):
        return self._size


"""
    def from_json(self, params):
        assert isinstance(params, dict), 'Invalid json input'

        for k, v in params.items():
            if isinstance(v, list):
                self.add({k: [self.get_param(p) for p in v]})
            else:
                self.add({k: self.get_param(v)})

    @staticmethod
    def get_param(p):
        assert isinstance(p, dict), ''
        assert 'type' in p, 'Missing parameter type'

        if p['type'] == 'int':
            assert 'min_value' in p, ''
            assert 'max_value' in p, ''
            return IntegerParameter(min_value=p['min_value'], max_value=p['max_value'])
        elif p['type'] == 'real':
            assert 'min_value' in p, ''
            assert 'max_value' in p, ''
            return RealParameter(min_value=p['min_value'], max_value=p['max_value'])
        elif p['type'] == 'categ':
            assert 'values' in p, ''
            return CategoricalParameter(values=p['values'])
        else:
            raise SyntaxError('Invalid parameter type - %s' % p['type'])
"""