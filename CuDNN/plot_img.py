import numpy as np
import matplotlib.pyplot as plt

gdt = np.loadtxt("gdt.txt", dtype=int).reshape(112,112)
pred = np.loadtxt("pred.txt", dtype=float).reshape(112,112)

plt.figure()
plt.subplot(121)
plt.imshow(pred )
plt.title('Prediction')
plt.subplot(122)
plt.imshow( gdt)
plt.title('Ground trouth')
plt.show()
