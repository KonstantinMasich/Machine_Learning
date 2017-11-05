import random
import matplotlib.pyplot as plt
import numpy as np


print(random.gauss(0, 1))
"""
nums = []
for i in range(0, 100):
    nums.append(random.gauss(0, 0.1))
nums = list(set(nums))
plt.hist(nums, bins = 100)
plt.show()
"""
#w = np.empty(shape=(3, 4))
#w.fill(random.gauss(0, 1))
#print(w)


mu = 0
sigma = 0.1
s = np.random.normal(mu, sigma, size=(3))
print(s)
#plt.hist(s, bins = 1000)
#plt.show()
"""
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()
"""
