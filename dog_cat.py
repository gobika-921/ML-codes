import numpy as np 
import matplotlib.pyplot as plt


dog = 500
cat = 500

dog_height = 28 + 4 * np.random.randn(dog)
cat_height = 24 + 4 * np.random.randn(cat)

plt.hist([dog_height, cat_height], stacked=True, color=['blue','orange'])
plt.show()
