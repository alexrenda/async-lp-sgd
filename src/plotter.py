#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import sys
import fileinput
import collections
import os

figures = [
    'Gradient size vs iteration count',
    'Gradient size vs wall clock time',
    'Distance to optimum vs iteration count',
    'Distance to optimum vs wall clock time',
    'Training loss vs iteration count',
    'Training loss vs wall clock time',
    'Errors vs iteration count',
    'Errors vs wall clock time',
]

def main():
    itcounts = collections.defaultdict(int)
    iterations = collections.defaultdict(list)
    fields = collections.defaultdict(list)

    for line in fileinput.input():
        fname = fileinput.filename()
        itcounts[fname] += 1
        i = itcounts[fname]

        iterations[fname].append(i)
        fields[fname].append(map(float, line.split()))

    all_iterations = {f: np.array(iters) for (f, iters) in iterations.items()}
    all_vals = {f: list(map(np.array, zip(*fs))) for (f, fs) in fields.items()}

    for figidx, title in enumerate(figures):
        plt.figure(figidx)
        plt.title(title)

    for fname in all_iterations:
        iterations = all_iterations[fname]
        vals = all_vals[fname]
        lname = os.path.basename(fname)

        times = vals[0]

        try:
            grad_sizes = vals[1]

            plt.figure(0)
            plt.semilogy(iterations, grad_sizes, label='{} gradient norm'.format(lname))

            plt.figure(1)
            plt.semilogy(times, grad_sizes, label='{} gradient norm'.format(lname))

            dist_to_opt = vals[2]

            plt.figure(2)
            plt.semilogy(iterations, grad_sizes, label='{} distance to optimum'.format(lname))

            plt.figure(3)
            plt.semilogy(times, grad_sizes, label='{} distance to optimum'.format(lname))


            train_losses = vals[3]

            plt.figure(4)
            plt.semilogy(iterations, train_losses, label='{} training loss'.format(lname))
            plt.figure(5)
            plt.semilogy(times, train_losses, label='{} training loss'.format(lname))


            train_errors = vals[4]
            test_errors = vals[5]

            plt.figure(6)
            plt.plot(iterations, train_errors, label='{} training error'.format(lname))
            plt.plot(iterations, test_errors, label='{} testing error'.format(lname))

            plt.figure(7)
            plt.plot(times, train_errors, label='{} training error'.format(lname))
            plt.plot(times, test_errors, label='{} testing error'.format(lname))
        except:
            continue

    for figidx, title in enumerate(figures):
        plt.figure(figidx)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
