import math as m


class Node:
    def __init__(self, value):
        self._value = value
        self._size = calc_size(value)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    def get_value(self, x):
        assert len(x) >= self._size, 'x must contains at least %d elements!' % self._size

        if isinstance(self._value, dict):
            ret_params = {}
            param_idx = 0
            for k, v in self._value.items():
                v_size = v.size
                ret_params[k] = v.get_value(x[param_idx:(param_idx+v_size)])
                param_idx += v_size
            return ret_params

        # literals (i.e. int, float, string...)
        return self._value


class ChoiceNode(Node):
    def __init__(self, value):
        self._value = [nodefy(v) for v in value]
        self._size = max([v.size for v in self._value]) + 1

    def get_value(self, x):
        n = self._value[int(round(x[0] * (len(self._value)-1)))]
        return n.get_value(x[1:])


class IntParameterNode(Node):
    def __init__(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self._size = 1

    def get_value(self, x):
        return m.floor(self._min + x[0] * (self._max - self._min))


class FloatParameterNode(Node):
    def __init__(self, min_val, max_val):
        self._min = min_val
        self._max = max_val
        self._size = 1

    def get_value(self, x):
        return self._min + x[0] * (self._max - self._min)


class ListNode(Node):
    def __init__(self, value):
        self._value = value
        self._size = sum([v.size for v in self._value])

    def get_value(self, x):
        ret_params = []
        param_idx = 0
        for v in self._value:
            v_size = v.size
            ret_params.append(v.get_value(x[param_idx:(param_idx+v_size)]))
            param_idx += v_size
        return ret_params


"""
    Helpers
"""
def hp_space(values):
    assert isinstance(values, dict), 'hp_space expects a dictionary!'
    return nodefy(values)


def hp_choice(options):
    assert isinstance(options, list), 'options must be a list!'
    assert len(options) > 0, 'options cannot be empty!'
    return ChoiceNode(options)


def hp_int(min_value, max_value):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return IntParameterNode(min_value, max_value)


def hp_float(min_value, max_value):
    assert min_value < max_value, 'max_value must be greater than min_value'
    return FloatParameterNode(min_value, max_value)


def nodefy(value):
    if isinstance(value, dict) and len(value) > 0:
        return Node({k: nodefy(v) for k, v in value.items()})

    elif isinstance(value, list) and len(value) > 0:
        return ListNode([nodefy(v) for v in value])

    elif isinstance(value, Node):
        return value

    return Node(value)


def calc_size(space):
    size = 0
    if isinstance(space, dict) and len(space) > 0:
        size = sum([calc_size(v) for v in space.values()])

    elif isinstance(space, list) and len(space) > 0:
        size = sum([calc_size(v) for v in space])

    elif isinstance(space, Node):
        return space.size

    return size
