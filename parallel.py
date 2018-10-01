
import random
from multiprocessing import Pool
import time

def list_append(count):
    out_list = []
    for i in range(count):
        out_list.append(random.random())


def parallel1():
    size = 10000000
    threads = 2
    jobs = []
    for i in range(0, threads):
        out_list = list()
        thread = threading.Thread(target=list_append(size, i, out_list))
        jobs.append(thread)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    print ("List processing complete.")
if __name__ == "__main__":
    start = time.time()
    with Pool(1) as p:
        print(p.map(list_append, [1000000, 2000000, 3000000]))
    print("Time taken = {0:.5f}".format(time.time() - start))
