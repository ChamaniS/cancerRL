import matplotlib.pyplot as plt
import numpy as np

images = np.random.uniform(0, 255, size=(40, 50, 50))

fig, ax = plt.subplots()

im = ax.imshow(images[0])
im = ax.imshow(images[0])
fig.show()
for image in images[1:]:
    im.set_data(image)
    #fig.canvas.draw()
    plt.pause(0.5)