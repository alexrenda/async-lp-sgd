#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import sys
import fileinput
import collections
import os

ic_str = 'Iteration count'
wct_str = 'Wall clock time (s)'

figures = [
    ('Training loss', ic_str),
    ('Training loss', wct_str),
    ('Test error', ic_str),
    ('Test error', wct_str),
]

def main():
    itcounts = collections.defaultdict(int)
    iterations = collections.defaultdict(list)
    fields = collections.defaultdict(list)

    for line in fileinput.input():
        fname = fileinput.filename()
        itcounts[fname] += 1
        i = itcounts[fname]

        if (i % 100 == 0):
            iterations[fname].append(i)
            fields[fname].append(map(float, line.split()))

    all_iterations = {f: np.array(iters) for (f, iters) in iterations.items()}
    all_vals = {f: list(map(np.array, zip(*fs))) for (f, fs) in fields.items()}

    for figidx, (yaxis, xaxis) in enumerate(figures):
        plt.figure(figidx)
        plt.title('{} vs {}'.format(yaxis, xaxis))
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

    for fname in all_iterations:
        iterations = all_iterations[fname]
        vals = all_vals[fname]
        lname = os.path.basename(fname)

        times = vals[0]

        try:
            train_losses = vals[1]

            plt.figure(0)
            plt.semilogy(iterations, train_losses, label='{} training loss'.format(lname))
            plt.figure(1)
            plt.semilogy(times, train_losses, label='{} training loss'.format(lname))


            train_errors = vals[2]
            test_errors = vals[3]

            plt.figure(2)
            plt.plot(iterations, test_errors, label='{} testing error'.format(lname))

            plt.figure(3)
            plt.plot(times, test_errors, label='{} testing error'.format(lname))
        except:
            continue

    for figidx, title in enumerate(figures):
        plt.figure(figidx)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
