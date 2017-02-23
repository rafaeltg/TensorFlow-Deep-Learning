import math as m


class Node:
    def __init__(self, value, label=''):
        self._label = label
        self._value = value
        self._size = self._get_size()

    def _get_size(self):
        return calc_size(self._value)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    def get_value(self, x):
        assert len(x) >= self._size, 'x must contains at least %d elements!' % self._size

        if isinstance(self._value, list):
            ret_params = {}
            param_idx = 0
            for v in self._value:
                v_size = v.size
                ret_params[v.label] = v.get_value(x[param_idx:(param_idx+v_size)])
                param_idx += v_size
            return ret_params

        # literals (i.e. int, float, string...)
        return self._value


class ChoiceNode(Node):
    def __init__(self, value, label=''):
        super().__init__(value=[nodefy(v) for v in value], label=label)

    def _get_size(self):
        return max([v.size for v in self._value]) + 1

    def get_value(self, x):
        n = self._value[int(round(x[0] * (len(self._value)-1)))]
        return n.get_value(x[1:])


class IntParameterNode(Node):
    def __init__(self, min_val, max_val, label=''):
        super().__init__(None, label)
        self._min = min_val
        self._max = max_val

    def _get_size(self):
        return 1

    def get_value(self, x):
        return m.floor(self._min + x[0] * (self._max - self._min))


class FloatParameterNode(Node):
    def __init__(self, min_val, max_val, label=''):
        super().__init__(None, label)
        self._min = min_val
        self._max = max_val

    def _get_size(self):
        return 1

    def get_value(self, x):
        return self._min + x[0] * (self._max - self._min)


class ListNode(Node):
    def __init__(self, value, label=''):
        super().__init__(value, label)

    def _get_size(self):
        return sum([v.size for v in self._value])

    def get_value(self, x):
        ret_params = []
        param_idx = 0
        for v in self._value:
            v_size = v.size
            ret_params.append(v.get_value(x[param_idx:(param_idx+v_size)]))
            param_idx += v_size
        return ret_params


#
# HELPERS
#

def hp_space(values):
    assert isinstance(values, dict), 'values must be a dictionary!'
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
        values = []
        for k, v in value.items():
            n = nodefy(v)
            n.label = k
            values.append(n)
        return Node(sorted(values, key=lambda node: node.label))

    elif isinstance(value, list) and len(value) > 0:
        return ListNode([nodefy(v) for v in value])

    elif isinstance(value, Node):
        return value

    return Node(value)


def calc_size(space):
    size = 0
    if isinstance(space, list) and len(space) > 0:
        size = sum([calc_size(v) for v in space])

    elif isinstance(space, Node):
        return space.size

    return size


def hp_space_from_json(values):

    if isinstance(values, dict):
        if "node_type" in values:
            if values['node_type'] == 'hp_choice':
                return ChoiceNode([hp_space_from_json(v) for v in values['value']])

            elif values['node_type'] == 'hp_list':
                return ListNode([hp_space_from_json(v) for v in values['value']])

            elif values['node_type'] == 'hp_int':
                return IntParameterNode(values['min_val'], values['max_val'])

            elif values['node_type'] == 'hp_float':
                return FloatParameterNode(values['min_val'], values['max_val'])
        else:
            for k, v in values.items():
                values[k] = hp_space_from_json(v)

    return nodefy(values)
