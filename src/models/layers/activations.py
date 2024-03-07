from torch import nn


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if hasattr(nn, identifier):
            cls = getattr(nn, identifier)
        else:
            cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret activation identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(identifier))
