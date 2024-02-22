# โครงสร้าง 3 input nodes, 2 hidden nodes, 1 output node
# มี input ชุดเดียว คำนวณ backprop 1 ครั้งเพื่อปรับค่า
# import numpy
from NNfunction import *  # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
import random
import pandas as pd


def forward_propagation(x, w, n):
    o = Nout(x, w)  # call NN function
    y = sigmoid(o)  # call NN function
    print(f"\nSum(V) of node {n} is: %8.3f, Y from node {n} is: %8.3f" % (o, y))
    return y


def backpropagation(g, learning_rate, x, w):
    b = []
    for i in range(len(x)):
        b.append(w[i] + (deltaw(learning_rate, g, x[i])))
    return b


def approximate(x, w10, w11, w12, w13, desire_output, learning_rate):
    print("\n-----Forward pass----->")
    # forward pass
    y10 = forward_propagation(x, w10, 10)

    y11 = forward_propagation(x, w11, 11)

    z = [y10, y11, 1]
    y12 = forward_propagation(z, w12, 12)

    y13 = forward_propagation(z, w13, 13)

    e12 = desire_output - y12
    e13 = desire_output - y13
    print("\nError of node 12 is: %8.3f, Error of node 13 is: %8.3f" % (e12, e13))
    e = (e12 + e13) / 2

    if e != 0:
        print("\n-----Backward pass----->")
        g13 = gradOut(e, y13)
        w13 = backpropagation(g13, learning_rate, z, w13)
        print("\nNew weights of node 13 are: %8.3f, %8.3f, %8.3f" % (w13[0], w13[1], w13[2]))

        g12 = gradOut(e, y12)
        w12 = backpropagation(g12, learning_rate, z, w12)
        print("\nNew weights of node 13 are: %8.3f, %8.3f, %8.3f" % (w12[0], w12[1], w12[2]))

        g11 = gradH(y11, (g13 * w13[1]) + (g12 * w12[1]))
        w11 = backpropagation(g11, learning_rate, x, w11)
        print("\nNew weights of node 12 are: %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f" % (
            w11[0], w11[1], w11[2], w11[3], w11[4], w11[5], w11[6], w11[7], w11[8], w11[9]))

        g10 = gradH(y10, (g13 * w13[0]) + (g12 * w12[0]))
        w10 = backpropagation(g10, learning_rate, x, w10)
        print("\nNew weights of node 11 are: %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f" % (
            w10[0], w10[1], w10[2], w10[3], w10[4], w10[5], w10[6], w10[7], w10[8], w10[9]))

    return e, w10, w11, w12, w13


def training_path(cycles):
    avg_error = 0
    learning_rate = -0.9
    w10 = [random.uniform(-1, 1) for _ in range(10)]
    w11 = [random.uniform(-1, 1) for _ in range(10)]
    w12 = [random.uniform(-1, 1) for _ in range(3)]
    w13 = [random.uniform(-1, 1) for _ in range(3)]

    dataset = pd.read_csv('train.csv')
    for j in range(cycles):
        avg_error = 0
        print(f"Round {j + 1}")
        for i in range(len(dataset)):
            data = dataset.iloc[i, 0:10].tolist()
            desire_output = 1 if data[9] == 2 else 0
            data[9] = 1
            e, w10, w11, w12, w13 = approximate(data, w10, w11, w12, w13, desire_output, learning_rate)
            avg_error += e
        avg_error /= len(dataset)

    print("\n-------------------End of Training------------------->")
    print("\nAverage Error: ", avg_error)
    print("\nTotal Cycles: ", cycles)
    print("\nFinal weights of node 10 are: ", w10)
    print("\nFinal weights of node 11 are: ", w11)
    print("\nFinal weights of node 12 are: ", w12)
    print("\nFinal weights of node 13 are: ", w13)

    return w10, w11, w12, w13, avg_error


def test_path(w10, w11, w12, w13, cycles, avg_error):
    data = pd.read_csv("test.csv")
    total = 0
    correct = 0
    for i in range(len(data)):
        x = data.iloc[i, 0:10].tolist()
        correct_ans = x[9]
        x[9] = 1
        # Forward
        print("\n-----Forward pass----->")
        # forward pass
        y10 = forward_propagation(x, w10, 10)

        y11 = forward_propagation(x, w11, 11)

        z = [y10, y11, 1]
        y12 = forward_propagation(z, w12, 12)

        y13 = forward_propagation(z, w13, 13)

        y12 = 1 if y12 > 0.5 else 0
        y13 = 1 if y13 > 0.5 else 0
        result = [y12, y13]
        result = 2 if result == [1, 1] else 4
        print("\nDesire Output: ", [1, 1] if x[9] == 2 else [0, 0])
        print("\nOutput: ", [y12, y13])
        print("Correct Answer: ", correct_ans, "Predicted Answer: ", result)
        print("Correct" if correct_ans == result else "Wrong")
        if correct_ans == result:
            correct += 1
        total += 1
        print("\n------------------------------------------------->")

    print("\n-------------------FINAL RESULT------------------->")
    print("This test with model trained with ", cycles, " epochs with avg error: ", avg_error)
    print("Total: ", total)
    print("Correct: ", correct)
    print("Wrong: ", total - correct)
    print("Accuracy: ", correct / total * 100, "%")


def calculate(cycles):
    w10, w11, w12, w13, avg_error = training_path(cycles)
    test_path(w10, w11, w12, w13, cycles, avg_error)
