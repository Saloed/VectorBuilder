import time
from traceback import print_tb


def timing(f):
    def wrap(*args):
        print('{} function start'.format(f.__name__, ))
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        elapse = (time2 - time1) * 1000.0
        seconds = elapse / 1000
        millis = elapse % 1000
        print('{} function elapse {} sec {} ms'.format(f.__name__, seconds, millis))
        return ret

    return wrap


def safe_run(f):
    def wrap(*args):
        try:
            ret = f(*args)
        except Exception as exc:
            print('exception in {}'.format(f.__name__))
            print(type(exc))  # the exception instance
            print(exc.args)  # arguments stored in .args
            print(exc)
            print_tb(exc.__traceback__)
            return None
        return ret

    return wrap
