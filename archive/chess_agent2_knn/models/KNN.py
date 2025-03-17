import numpy as np

class KNN:

    def __init__(self, states, input_arr, color):
        # color = 1, then white
        # black otherwise
        self.arr = []
        input_arr = np.array(input_arr)
        for state in states:
            arr1 = np.array(state)
            mae = np.mean(np.abs(arr1 - input_arr))
            self.arr.append((mae, states[state]))
        self.arr.sort(key=lambda x: x[0])

    def get_average_topk(self, k: int):
        if k > len(self.arr):
            raise ValueError("To many K selected")
        temp = [0.0] * 64
        for i in range(k):
            arr = self.arr[i][1]
            for j in range(len(arr)):
                temp[j] += arr[j]
        return temp