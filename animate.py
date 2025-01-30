import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image


def animate_flow(sim_flow, net_flow, barrier, output_gif_path="flow_animation.gif"):
    height, width = barrier.shape
    frames = []  # List to store each frame

    for i in range(len(net_flow)):
        vmin = np.min(sim_flow[i])  # Get the global minimum from sim_flow
        vmax = np.max(sim_flow[i])  # Get the global maximum from sim_flow

        plt.subplot(2, 1, 1)
        fluidImage = plt.imshow(
            net_flow[i],
            origin="lower",
            cmap=plt.get_cmap("jet"),
            interpolation="none",
            vmin=vmin,  # Set the same minimum
            vmax=vmax,  # Set the same maximum
        )
        bImageArray = np.zeros((height, width, 4), np.uint8)
        bImageArray[barrier, 3] = 255
        barrierImage = plt.imshow(bImageArray, origin="lower", interpolation="none")
        plt.colorbar(fluidImage)

        plt.subplot(2, 1, 2)
        fluidImage = plt.imshow(
            sim_flow[i],
            origin="lower",
            cmap=plt.get_cmap("jet"),
            interpolation="none",
            vmin=vmin,  # Ensure same min scale
            vmax=vmax,  # Ensure same max scale
        )
        bImageArray = np.zeros((height, width, 4), np.uint8)
        bImageArray[barrier, 3] = 255
        barrierImage = plt.imshow(bImageArray, origin="lower", interpolation="none")
        plt.colorbar(fluidImage)

        plt.tight_layout()

        # Save the current frame
        plt.draw()  # Draw the current figure
        img_buf = plt.gcf()  # Get the current figure
        img_buf.canvas.draw()  # Render the image
        img_array = np.array(
            img_buf.canvas.renderer.buffer_rgba()
        )  # Get image as RGBA array

        # Convert to PIL Image and add to frames
        pil_img = Image.fromarray(img_array)
        frames.append(pil_img)

        plt.clf()  # Clear the figure after each frame

    # Save all frames as a GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],  # Append remaining frames
        duration=200,  # Duration of each frame (in milliseconds)
        loop=0,  # 0 means the animation loops infinitely
    )

    print(f"GIF saved as {output_gif_path}")
