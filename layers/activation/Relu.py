from layers.layer import Layer


class Relu(Layer):
    def __init__(self):
        self.mask = None
        super().__init__()
        
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
