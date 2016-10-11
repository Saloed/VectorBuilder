from NN.Layer import Layer


def print_layers(layers: list):
    def print_layer(layer: Layer, shift="\t"):
        print(shift, layer.name)
        for conn in layer.in_connection:
            print_layer(conn.from_layer, shift + "\t")

    print_layer(layers[0])
