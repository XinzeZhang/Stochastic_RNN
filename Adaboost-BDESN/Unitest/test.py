import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(start=-10, stop=10, num=101)
plt.plot(x, np.absolute(x))
plt.show()