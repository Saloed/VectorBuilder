import _pickle as P
import numpy as np


def main():
    np.set_printoptions(threshold=1000000)
    filename = 'NewParams/new_params_t0_ep23'
    with open(filename, 'rb') as f:
        params = P.load(f)

    with open('params.txt', 'w') as out:
        for w in list(params.w.values()):
            print(w, file=out)
            print(w.eval(), file=out)
        for b in list(params.b.values()):
            print(b, file=out)
            print(b.eval(), file=out)


if __name__ == '__main__':
    main()
