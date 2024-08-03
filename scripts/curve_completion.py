# curve_completion.py

import numpy as np
from scipy.interpolate import splprep, splev
# from utils import read_csv, plot

def complete_curve(path_XYs):
    completed_paths = []
    for shape in path_XYs:
        completed_shape = interpolate_missing_parts(shape)
        completed_paths.append(completed_shape)
    return completed_paths

def interpolate_missing_parts(XYs):
    tck, u = splprep([XYs[:, 0], XYs[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), len(XYs))
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

# if __name__ == "__main__":
#     path_XYs = read_csv('/content/drive/MyDrive/curve_project/data/occlusion1.csv')
#     completed_paths = complete_curve(path_XYs)
#     plot(completed_paths, title='Completed Curves')


