+from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math as m


class ParameterDictionary(object):

    def __init__(self):
        self.params = {}
        self.size = 0

    def add(self, param):
        assert isinstance(param, dict), '"param" must be dictionary'

        for k, v in param.items():
            if isinstance(v, BaseParameter):
                self.size += 1
            elif isinstance(v, list):
                assert all([isinstance(p, BaseParameter) for p in v]), ''
                self.size += len(v)
            else:
                raise TypeError("Test")

            self.params[k] = v

    def get(self, x):
        assert len(x) == self.size, 'Invalid required number of parameters'
        ret_params = {}
        i = 0

        for k, v in self.params.items():
            if isinstance(v, list):
                p_list = []
                for p in v:
                    p_list.append(p.get_value(x[i]))
                    i += 1
                ret_params[k] = p_list
            else:
                ret_params[k] = v.get_value(x[i])
                i += 1

        return ret_params

    def from_json(self, dic):
        pass


class BaseParameter(object):

    def get_value(self, idx):
        pass


class ListParameter(BaseParameter):

    def __init__(self, values):
        self.values = values

    def get_value(self, idx):
        assert 0 <= idx <= 1, 'Parameter "idx" must be greater or equal to zero and less or equal to 1.'
        return self.values[round(idx * (len(self.values)-1))]


class IntegerParameter(BaseParameter):

    def __init__(self, min_value, max_value, transform=None):
        self.min_value = min_value
        self.max_value = max_value
        self.transform = transform

    def get_value(self, idx):
        assert 0 <= idx <= 1, 'Parameter "idx" must be greater or equal to zero and less or equal to 1.'

        if self.transform is None:
            return m.floor(self.min_value + idx * (self.max_value - self.min_value))
        else:
            return self.transform(idx)


class RealParameter(BaseParameter):

    def __init__(self, min_value, max_value, transform=None):
        self.min_value = min_value
        self.max_value = max_value
        self.transform = transform

    def get_value(self, idx):
        assert 0 <= idx <= 1, 'Parameter "idx" must be greater or equal to zero and less or equal to 1.'

        if self.transform is None:
            return idx
        else:
            return self.transform(idx)
