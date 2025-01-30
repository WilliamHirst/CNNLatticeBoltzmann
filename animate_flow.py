import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import (
    display,
    HTML,
)  # To ensure proper inline display of animation


def main(stream, collide, curl, box, barrier, init_dense):
    height, width = box
    ux, uy = init_dense
    # Visualization
    fig = plt.figure(figsize=(8, 3))
    fluidImage = plt.imshow(
        curl(ux, uy),
        origin="lower",
        norm=plt.Normalize(-0.1, 0.1),
        cmap=plt.get_cmap("jet"),
        interpolation="none",
    )
    bImageArray = np.zeros((height, width, 4), np.uint8)
    bImageArray[barrier, 3] = 255
    barrierImage = plt.imshow(bImageArray, origin="lower", interpolation="none")

    # Frame update function
    def next_frame(arg):
        for _ in range(25):
            stream()
            collide()
        fluidImage.set_array(curl(ux, uy))
        return fluidImage, barrierImage

    # Run animation
    ani = matplotlib.animation.FuncAnimation(fig, next_frame, frames=3200)
    HTML(ani.to_jshtml())
