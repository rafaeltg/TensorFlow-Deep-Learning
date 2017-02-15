from ..base import SupervisedModel


class MLP(SupervisedModel):

    """ Multi-Layer Perceptron
    """

    def __init__(self,
                 name='mlp',
                 layers=None,
                 **kwargs):

        """
        :param layers: List of layers in the network.
        """

        super().__init__(name=name,
                         layers=layers,
                         **kwargs)

        self.logger.info('Done {} __init__'.format(__class__.__name__))
