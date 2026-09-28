
def adj_open_to_open_h(frame, horizon = 1):
    return frame["adj_open"].shift(-horizon) / frame["adj_open"] - 1
