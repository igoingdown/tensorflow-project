from threading import *


def counter_1(x, label=""):
    for k in xrange(x):
        print k, "\t", label


if __name__ == "__main__":
    thread_pool = []
    for i in range(2):
        thread_pool.append(Thread(group=None, name="t1", target=counter_1, args=(10, "t1")))
        thread_pool.append(Thread(group=None, name="t2", target=counter_1, args=(20, "t2")))

    for t in thread_pool:
        t.run()

