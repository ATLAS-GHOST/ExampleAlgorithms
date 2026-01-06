from tensorflow.keras import Input, Model, backend, layers, initializers

from modules.layers import (
    SlidingConeSum,
    LocalMaxMask,
    ImageToMomentumList
)

def CNNJetAlgo(
        kernel_size : int = 7,
        channels    : int = 1,
        n_conv      : int = 16,
        extra_conv2d: list = None,
        extra_dense : list = None,
        fix_layer   : bool = False,
        head        : str = ""
):
    inp = Input(shape=(56,70,channels))

    #Base Layer Conv2D
    x = layers.Conv2D(n_conv, kernel_size=kernel_size, activation='relu', padding='same')(inp)

    #Extra Conv2D (if any)
    if extra_conv2d:
        for n_ker, n_size, activ in extra_conv2d:
            x = layers.Conv2D(n_ker, kernel_size=n_size, activation=activ, padding='same')(x)

    #Add Pooling (not needed)
    if head == "gap":
        x = layers.GlobalAveragePooling2D()(x)
    elif head == "flatten":
        x = layers.Flatten()(x)

    if fix_layer:
        x = layers.Conv2D(64, kernel_size=1, activation='relu', padding='same')(x)
        x = layers.Conv2D(32, kernel_size=1, activation='relu', padding='same')(x)

        if extra_dense:
            for n_nodes, activ in extra_dense:
                x = layers.Conv2D(n_nodes, kernel_size=1, activation=activ, padding='same')(x)

        x     = layers.Conv2D(1, kernel_size=1, activation='relu')(x)
        seeds = x

        coneSum = SlidingConeSum(kernel_size=3,
                                 shape="square",
                                 radius=None,
                                 name="cone")
        x = coneSum(x)
        x = LocalMaxMask(kernel_size=3,
                         name="local_max")([x, seeds])
        
    else:
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)

        if extra_dense:
            for n_nodes, activ in extra_dense:
                x = layers.Dense(n_nodes, activation=activ)(x)

        x = layers.Dense(1, activation='relu')(x)

    return Model(inp, x, name='cnn_jet_algo')
        
