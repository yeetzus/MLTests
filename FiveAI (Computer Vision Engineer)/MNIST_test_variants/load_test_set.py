import numpy as np

if __name__ == "__main__":

    for tst in ["clean", "t1", "t2", "t3", "t4"]:
        data = np.load("test_sets/" + tst + ".npy", allow_pickle=True).item()
        x, y = data['x'], data['y']
        print(tst, '\t', x.shape, '\t', y.shape)


