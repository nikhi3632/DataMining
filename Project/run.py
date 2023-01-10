#!/usr/bin/env python3

from clustering import main
from time import perf_counter

if __name__ == "__main__":
    t1_start = perf_counter()
    main()
    t1_stop = perf_counter()
    print("Elapsed time during the whole program in seconds = %.2f" %(t1_stop-t1_start))