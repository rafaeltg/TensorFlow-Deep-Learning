from .utils import get_input_data, load_data, get_model
from pydl.models import SupervisedModel
from pydl.models.utils.utilities import load_model, save_model


def fit(config, output):
    """
    """

    m = load_model(get_model(config))

    data_set = get_input_data(config)
    x = load_data(data_set, 'train_x')

    if isinstance(m, SupervisedModel):
        y = load_data(data_set, 'train_y')
        m.fit(x_train=x, y_train=y)
    else:
        m.fit(x_train=x)

    # Save model
    save_model(model=m, dir=output)
