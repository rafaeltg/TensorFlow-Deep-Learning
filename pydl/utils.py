import os
import tempfile
import six
import numpy as np


__all__ = ['dump_np_array', 'free_mmap_np_array']


def dump_np_array(data, filename='', dir='/dev/shm', mode='r'):

    if isinstance(data, np.core.memmap):
        return data

    if filename == '':
        mmap_file = tempfile.NamedTemporaryFile(dir=dir).name
    else:
        mmap_file = os.path.join(dir, filename)
        if os.path.exists(mmap_file):
            os.unlink(mmap_file)

    fp = np.memmap(mmap_file, dtype='float64', mode='w+', shape=data.shape)
    fp[:] = data[:]
    del fp
    return np.memmap(mmap_file, dtype='float64', mode=mode, shape=data.shape)


def dump_np_data_set(x, y=None):
    _x = dump_np_array(x)
    _y = dump_np_array(y) if y is not None else None
    return _x, _y


def free_mmap_np_array(fp):
    file_name = ''
    try:
        if isinstance(fp, np.core.memmap):
            file_name = fp.filename
        elif isinstance(fp, six.string_types):
            file_name = fp

        if file_name != '' and os.path.exists(file_name):
            os.unlink(file_name)

    except Exception as e:
        print('Fail to remove %s (%s)' % (file_name, repr(e)))


def free_mmap_data_set(x, y=None):
    free_mmap_np_array(x)

    if y is not None:
        free_mmap_np_array(y)
