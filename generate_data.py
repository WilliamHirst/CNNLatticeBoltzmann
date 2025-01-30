import numpy as np
import random


class GenerateFlowData:
    def __init__(self, box, speed_range, visocity_range):
        self.height, self.width = box
        self.min_speed, self.max_speed = speed_range
        self.min_viscosity, self.max_viscosity = visocity_range
        self.barrier = np.zeros((self.height, self.width), bool)
        self.shape_type = "line"  # ['line', 'square', 'triangle', 'circle']

        self.four9ths = 4.0 / 9.0
        self.one9th = 1.0 / 9.0
        self.one36th = 1.0 / 36.0
        self.timeframes = 200
        self.num_samples = 32

        self.stream_direction = {
            "N": (0, 1),  # "N" corresponds to roll in the x-axis (rows)
            "S": (0, -1),  # "S" corresponds to roll in the x-axis (rows)
            "E": (1, 1),  # "E" corresponds to roll in the y-axis (columns)
            "W": (1, -1),  # "W" corresponds to roll in the y-axis (columns)
            "NE": (0, 1),  # "NE" corresponds to both x and y axes
            "NW": (0, -1),  # "NW" corresponds to both x and y axes
            "SE": (0, 1),  # "SE" corresponds to both x and y axes
            "SW": (0, -1),  # "SW" corresponds to both x and y axes
        }
        self.barrier_bounces = {
            "N": "S",
            "S": "N",
            "E": "W",
            "W": "E",
            "NE": "SW",
            "NW": "SE",
            "SE": "NW",
            "SW": "NE",
        }

    def get_dataset(self):
        X = np.zeros(
            (self.num_samples, self.timeframes, self.height, self.width, 4)
        )  # (samples, timeframes, height, width, 4)
        Y = np.zeros(
            (self.num_samples, self.timeframes, self.height, self.width, 2)
        )  # (samples, timeframes, height, width, 2)

        for i in range(self.num_samples):
            X_sample, Y_sample = self.gen_X_and_Y()
            X[i] = X_sample
            Y[i] = Y_sample
        return X, Y

    def gen_X_and_Y(self):
        # Generate random barrier and direction
        barrier, barrier_direction = self.generate_random_barrier()

        # Randomly select viscosity and flow speed
        u0 = random.uniform(self.min_speed, self.max_speed)
        visc = random.uniform(self.min_viscosity, self.max_viscosity)

        viscosity = np.full(
            (self.height, self.width),
            visc,
        )
        speed = np.full((self.height, self.width), u0)

        # Generate timestep layer (constant across the grid)
        timestep_layer = np.full(
            (self.height, self.width), 0
        )  # Assuming timestep 0 for the initial state

        # Stack the input features (barrier, viscosity, speed, timestep) along the last axis
        X = np.stack(
            [barrier, viscosity, speed, timestep_layer], axis=-1
        )  # Shape: (height, width, 4)

        # Calculate omega from viscosity
        omega = 1 / (3 * visc + 0.5)

        # Generate the simulation results (density and velocity fields) for each time step
        Y = self.sample(barrier, barrier_direction, omega, u0)

        # Stack the output fields (velocity and density fields) for each time step
        return X, Y

    def sample(self, barrier, barrier_direction, omega, u0):
        n, rho, ux, uy = self.initialize_arrays(u0)
        Curl = np.zeros((self.timeframes, self.height, self.width))
        for i in range(self.timeframes * 10):
            n = self.stream(n, barrier, barrier_direction)
            n, rho, ux, uy = self.collide(n, rho, ux, uy, omega, u0)
            Curl[int(i / 10)] = self.curl(ux, uy)
            if i % 10 == 0 and i:
                exit()
        return Curl

    # Stream function
    def stream(self, n, barrier, barrier_direction):
        for key, (axis, shift) in self.stream_direction.items():
            n[key] = np.roll(n[key], shift, axis=axis)

        # Bounce-back boundary conditions
        for key, opposite_key in self.barrier_bounces.items():
            n[key][barrier_direction[key]] = n[opposite_key][barrier]
        return n

    # Collide function
    def collide(self, n, rho, ux, uy, omega, u0):
        print("ux", ux)
        print("uy", uy)
        # Calculate rho, ux, uy, and other parameters
        rho = np.sum(
            n[dir] for dir in ["0", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]
        )
        rho = np.maximum(rho, 1e-3)  # Ensures no division by zero

        ux = (n["E"] + n["NE"] + n["SE"] - n["W"] - n["NW"] - n["SW"]) / rho
        uy = (n["N"] + n["NE"] + n["NW"] - n["S"] - n["SE"] - n["SW"]) / rho
        ux = np.clip(ux, -1.0, 1.0)
        uy = np.clip(uy, -1.0, 1.0)
        ux2 = ux * ux
        uy2 = uy * uy
        u2 = ux2 + uy2
        u02 = u0**2
        u_ineer = 3 * u0 + 3 * u02

        omu215 = 1 - 1.5 * u2
        uxuy = ux * uy

        # Update the n variables using the Bhatnagar–Gross–Krook (BGK) update rule
        n["0"] = (1 - omega) * n["0"] + omega * self.four9ths * rho * omu215
        n["N"] = (1 - omega) * n["N"] + omega * self.one9th * rho * (
            omu215 + 3 * uy + 4.5 * uy2
        )
        n["S"] = (1 - omega) * n["S"] + omega * self.one9th * rho * (
            omu215 - 3 * uy + 4.5 * uy2
        )
        n["E"] = (1 - omega) * n["E"] + omega * self.one9th * rho * (
            omu215 + 3 * ux + 4.5 * ux2
        )
        n["W"] = (1 - omega) * n["W"] + omega * self.one9th * rho * (
            omu215 - 3 * ux + 4.5 * ux2
        )
        n["NE"] = (1 - omega) * n["NE"] + omega * self.one36th * rho * (
            omu215 + 3 * (ux + uy) + 4.5 * (u2 + 2 * uxuy)
        )
        n["NW"] = (1 - omega) * n["NW"] + omega * self.one36th * rho * (
            omu215 + 3 * (-ux + uy) + 4.5 * (u2 - 2 * uxuy)
        )
        n["SE"] = (1 - omega) * n["SE"] + omega * self.one36th * rho * (
            omu215 + 3 * (ux - uy) + 4.5 * (u2 - 2 * uxuy)
        )
        n["SW"] = (1 - omega) * n["SW"] + omega * self.one36th * rho * (
            omu215 + 3 * (-ux - uy) + 4.5 * (u2 + 2 * uxuy)
        )
        for key in n.keys():
            n[key] = np.clip(n[key], -1.0, 1.0)

        # Apply boundary conditions for steady flow at the boundaries (force)
        n["E"][:, 0] = self.one9th * (1 + u_ineer)
        n["W"][:, 0] = self.one9th * (1 - u_ineer)
        n["NE"][:, 0] = self.one36th * (1 + u_ineer)
        n["SE"][:, 0] = self.one36th * (1 + u_ineer)
        n["NW"][:, 0] = self.one36th * (1 - u_ineer)
        n["SW"][:, 0] = self.one36th * (1 - u_ineer)

        return n, rho, ux, uy

    # Calculate curl
    def curl(self, ux, uy):
        return (
            np.roll(uy, -1, axis=1)
            - np.roll(uy, 1, axis=1)
            - np.roll(ux, -1, axis=0)
            + np.roll(ux, 1, axis=0)
        )

    def initialize_arrays(self, u0):
        ones = np.ones((self.height, self.width), dtype=np.float32)
        n = {}
        # Initialize arrays
        u02 = u0**2
        n["0"] = self.four9ths * (ones - 1.5 * u02)
        n["N"] = self.one9th * (ones - 1.5 * u02)
        n["S"] = self.one9th * (ones - 1.5 * u02)
        n["E"] = self.one9th * (ones + 3 * u0 + 3.0 * u02)
        n["W"] = self.one9th * (ones - 3 * u0 + 3.0 * u02)
        n["NE"] = self.one36th * (ones + 3 * u0 + 3.0 * u02)
        n["SE"] = self.one36th * (ones + 3 * u0 + 3.0 * u02)
        n["NW"] = self.one36th * (ones - 3 * u0 + 3.0 * u02)
        n["SW"] = self.one36th * (ones - 3 * u0 + 3.0 * u02)
        rho = sum(n[dir] for dir in ["0", "N", "S", "E", "W", "NE", "NW", "SE", "SW"])
        ux = (n["E"] + n["NE"] + n["SE"] - n["W"] - n["NW"] - n["SW"]) / rho
        uy = (n["N"] + n["NE"] + n["NW"] - n["S"] - n["SE"] - n["SW"]) / rho
        return n, rho, ux, uy

    def generate_random_barrier(self):
        # Clear the grid (start with no obstacles)
        barrier = np.zeros((self.height, self.width), bool)
        """
        shape_type = self.shape_type
        # Choose a random shape type

            # Generate a random straight line
        length = np.random.normal(
            loc=10, scale=3
        )  # Random length (normal distribution)
        length = max(4, int(length))  # Ensure length is at least 1
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        if y + length <= self.height:
            barrier[y : y + length, x] = 1"""
        # self.height // 2
        barrier = np.zeros((self.height, self.width), bool)
        barrier[(self.height // 2) - 8 : (self.height // 2) + 8, 1] = True

        barrier_direction = {}

        barrier_direction["N"] = np.roll(barrier, 1, axis=0)
        barrier_direction["S"] = np.roll(barrier, -1, axis=0)
        barrier_direction["E"] = np.roll(barrier, 1, axis=1)
        barrier_direction["W"] = np.roll(barrier, -1, axis=1)
        barrier_direction["NE"] = np.roll(barrier_direction["N"], 1, axis=1)
        barrier_direction["NW"] = np.roll(barrier_direction["N"], -1, axis=1)
        barrier_direction["SE"] = np.roll(barrier_direction["S"], 1, axis=1)
        barrier_direction["SW"] = np.roll(barrier_direction["S"], -1, axis=1)
        self.barrier = barrier
        return barrier, barrier_direction


if __name__ == "__main__":
    gfd = GenerateFlowData((60, 120), (0.1, 0.1), (0.004, 0.005))

    x, y = gfd.gen_X_and_Y()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    for i in range(len(y)):
        fluidImage = plt.imshow(
            y[i],
            origin="lower",
            norm=plt.Normalize(-0.1, 0.1),
            cmap=plt.get_cmap("jet"),
            interpolation="none",
        )
        bImageArray = np.zeros((gfd.height, gfd.width, 4), np.uint8)
        bImageArray[gfd.barrier, 3] = 255
        barrierImage = plt.imshow(bImageArray, origin="lower", interpolation="none")

        plt.show()

    fig = plt.figure(figsize=(8, 3))

    fluidImage = plt.imshow(
        y[0],
        origin="lower",
        norm=plt.Normalize(-0.1, 0.1),
        cmap=plt.get_cmap("jet"),
        interpolation="none",
    )
    bImageArray = np.zeros((gfd.height, gfd.width, 4), np.uint8)
    bImageArray[gfd.barrier, 3] = 255
    barrierImage = plt.imshow(bImageArray, origin="lower", interpolation="none")

    def get_frame(i):
        return y[i], barrierImage

    ani = animation.FuncAnimation(fig, get_frame, frames=199)
    ani.save("fluid_animation.gif", writer="pillow", fps=30)

    plt.show()
