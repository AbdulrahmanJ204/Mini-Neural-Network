from layers.layer import Layer

# I made this class because first I built a Dense class that has Affine layer and activation Layer,
# but then I removed the Dense class.
class Linear(Layer):
    def forward(self, x):
        return x

    def backward(self, dout):
        return dout
