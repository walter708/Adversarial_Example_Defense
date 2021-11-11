from .base import DefenseMechanism


class Identity(DefenseMechanism):
    """
    This 'defense mechanism' is in fact none. It does not change the network
    applied to. It just lifts an unprotected network to the `DefenseMechanism`
    interface.
    """

    def __init__(self, network):
        """
        Parameters
        ----------
        network : networks.Network object
        """
        self.model = network

    def predict_n(self, *batches):
        yield tuple(self.model.predict(batch) for batch in batches)

    @staticmethod
    def parameter_names():
        return []
