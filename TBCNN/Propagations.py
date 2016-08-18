def forward_propagation(network: list):
    last_layer = network[0]

    f = open('out.txt', mode='w')

    for layer in network:
        last_layer = layer
        for con in layer.forward_connection:
            con.f_prop()

        print("\n" + layer.name + "\n", file=f)
        print(layer.forward(), file=f)
        print("\n###################\n", file=f)

    f.close()

    return last_layer.forward()
