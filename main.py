from Nn3I2H1Oex1forStu import *
from ucimlrepo import fetch_ucirepo
import pandas as pd
from Preprocess import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # data_before_preprocessing()
    # export_data_to_csv()
    x = list(range(1, 11))
    result = {}
    result1 = []
    result2 = []
    result3 = []
    result4 = []
    for i in range(10):
        result1.append(calculate(1))
        result2.append(calculate(5))
        result3.append(calculate(10))
        result4.append(calculate(25))

    print(result1)
    print(result2)
    print(result3)
    print(result4)
    plt.plot(x, result1, 'o', label=f"1 cycle")
    plt.plot(x, result2, '*', label=f"5 cycle")
    plt.plot(x, result3, '+', label=f"10 cycle")
    plt.plot(x, result4, 'D', label=f"25 cycle")

    plt.xlabel('Loop')
    plt.ylabel('Accuracy')
    plt.title('Comparing count of cycle')

    # Adding legend
    plt.legend()

    # Showing grid
    plt.grid(True)

    # Displaying the graph
    plt.show()
