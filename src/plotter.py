#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import sys
import fileinput

def main():
    iterations = []
    fields = []
    for i, line in enumerate(fileinput.input()):
        iterations.append(i)
        fields.append(map(float, line.split()))

    iterations = np.array(iterations)
    vals = list(map(np.array, zip(*fields)))

    times, grad_sizes = vals[:2]

    plt.figure()
    plt.title('Gradient size vs iteration count')
    plt.semilogy(iterations, grad_sizes, label='Gradient norm')
    plt.legend()

    plt.figure()
    plt.title('Gradient size vs wall clock time')
    plt.semilogy(times, grad_sizes, label='Gradient norm')
    plt.legend()

    if len(vals) > 2:
        train_losses, train_errors, test_errors = vals[2:]

        plt.figure()
        plt.title('Training loss vs iteration count')
        plt.semilogy(iterations, train_losses, label='Training loss')
        plt.legend()

        plt.figure()
        plt.title('Errors vs iteration count')
        plt.plot(iterations, train_errors, label='Training error')
        plt.plot(iterations, test_errors, label='Testing error')
        plt.legend()

        plt.figure()
        plt.title('Training loss vs wall clock time')
        plt.semilogy(times, train_losses, label='Training loss')
        plt.legend()

        plt.figure()
        plt.title('Errors vs wall clock time')
        plt.plot(times, train_errors, label='Training error')
        plt.plot(times, test_errors, label='Testing error')
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
