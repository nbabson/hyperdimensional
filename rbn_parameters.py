import subprocess
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def search():
    reps = 5
    with open('rbn_parameters.txt', 'w') as f:
        f.write('I\tK\t%\n')
        for i in range(15):
            for k in range(5):
                total = 0.0
                for _ in range(reps):
                    ans = subprocess.check_output(
                            'python hd_rbn.py --i ' + str(i) + ' --k ' + str(k),
                            shell=True)
                    total += float(ans.decode('utf8').split(' ')[-2])
                    print('\t%f' %(total))
                total = total / reps    
                print(total)    
                f.write('%i\t%i\t%f\n' %(i, k, total))

def plot():
    success = np.zeros(shape=(5, 15))
    with open('rbn_parameters.txt', 'r') as f:
        f.readline()
        for line in f:
            currentline = list(map(float, line.split('\t')))
            success[int(currentline[1])][int(currentline[0])] = \
                    currentline[2]

    x = np.linspace(0, 14, 15)
    y = np.linspace(0, 4, 5)
    X, Y = np.meshgrid(x, y)
    Z = success
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.contour3D(X, Y, Z, 50, cmap='binary')  # contour plot
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
            cmap='viridis', edgecolor='none') # surface plot
    ax.set_xlabel('I')
    ax.set_ylabel('K')
    ax.set_zlabel('Success')
    ax.view_init(60,35)
    plt.show()
 
if __name__ == '__main__':
    plot()
