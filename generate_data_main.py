import numpy as np
import random


class GenerateFlowData:
    def __init__(
        self,
        box,
        speed_range,
        visocity_range,
        use_curl=True,
        num_samples=32,
        num_timeframes=100,
    ):
        self.height, self.width = box
        self.min_speed, self.max_speed = speed_range
        self.min_viscosity, self.max_viscosity = visocity_range
        self.barrier = np.zeros((self.height, self.width), bool)
        self.shape_type = "line"  # ['line', 'square', 'triangle', 'circle']

        self.four9ths = 4.0 / 9.0
        self.one9th = 1.0 / 9.0
        self.one36th = 1.0 / 36.0
        self.timeframes = num_timeframes
        self.num_samples = num_samples
        self.num_channels = 4 if use_curl else 5
        self.use_curl = use_curl

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

    def get_dataset(self, normalize=True):
        X = np.zeros(
            (
                self.num_samples * self.timeframes,
                self.num_channels,
                self.height,
                self.width,
            )
        )  # (samples, timeframes, height, width, 4)
        Y = np.zeros(
            (
                self.num_samples * self.timeframes,
                1 if self.use_curl else 2,
                self.height,
                self.width,
            )
        )  # (samples, timeframes, height, width, 2)

        for i in range(self.num_samples):
            X_sample, Y_sample = self.gen_X_and_Y()
            X[i * self.timeframes : (i + 1) * self.timeframes] = X_sample
            Y[i * self.timeframes : (i + 1) * self.timeframes] = Y_sample
        if normalize:
            X, Y = self.normalize(X, Y)
        if self.use_curl:
            Y = Y[:, 0]
        return X, Y

    def gen_X_and_Y(self):
        # Generate random barrier and direction
        barrier = self.generate_random_barrier()
        # Assuming barrier is a NumPy array
        barrier = np.expand_dims(barrier, axis=[0, 1])  # equivalent to unsqueeze(0)
        barrier = np.repeat(barrier, self.timeframes, axis=0)

        # Randomly select viscosity and flow speped
        u0 = random.uniform(self.min_speed, self.max_speed)
        visc = random.uniform(self.min_viscosity, self.max_viscosity)

        # Create constant layers over timesteps
        viscosity = np.full((self.timeframes, 1, self.height, self.width), visc)
        speed = np.full((self.timeframes, 1, self.height, self.width), u0)

        # Calculate omega from viscosity
        omega = 1 / (3 * visc + 0.5)

        # Generate the simulation results (density and velocity fields) for each time step
        Y = self.sample(omega, u0)
        X = np.concatenate(
            [barrier, viscosity, speed, Y[:-1]], axis=1
        )  # Shape: (height, width, 4)
        Y = Y[1:]

        # Stack the output fields (velocity and density fields) for each time step
        return X, Y

    def sample(self, omega, u0):
        n, rho, ux, uy = self.initialize_arrays(u0)
        Curl = np.zeros((self.timeframes + 1, 1, self.height, self.width))
        U = np.zeros((self.timeframes + 1, 2, self.height, self.width))
        sample_freq = 80
        for i in range((self.timeframes + 1) * sample_freq):
            n = self.stream(n)
            ux, uy, rho = self.collide(omega, u0)
            if sample_freq % sample_freq == 0:
                if self.use_curl:
                    Curl[int(i / sample_freq)] = self.curl(ux, uy)
                else:
                    U[int(i / sample_freq), 0] = ux
                    U[int(i / sample_freq), 1] = uy
        if self.use_curl:
            return Curl
        else:
            return U

    # Stream function
    def stream(self, n):

        self.nN = np.roll(self.nN, 1, axis=0)
        self.nNE = np.roll(self.nNE, 1, axis=0)
        self.nNW = np.roll(self.nNW, 1, axis=0)
        self.nS = np.roll(self.nS, -1, axis=0)
        self.nSE = np.roll(self.nSE, -1, axis=0)
        self.nSW = np.roll(self.nSW, -1, axis=0)
        self.nE = np.roll(self.nE, 1, axis=1)
        self.nNE = np.roll(self.nNE, 1, axis=1)
        self.nSE = np.roll(self.nSE, 1, axis=1)
        self.nW = np.roll(self.nW, -1, axis=1)
        self.nNW = np.roll(self.nNW, -1, axis=1)
        self.nSW = np.roll(self.nSW, -1, axis=1)

        # Bounce-back boundary conditions
        self.nN[self.barrierN] = self.nS[self.barrier]
        self.nS[self.barrierS] = self.nN[self.barrier]
        self.nE[self.barrierE] = self.nW[self.barrier]
        self.nW[self.barrierW] = self.nE[self.barrier]
        self.nNE[self.barrierNE] = self.nSW[self.barrier]
        self.nNW[self.barrierNW] = self.nSE[self.barrier]
        self.nSE[self.barrierSE] = self.nNW[self.barrier]
        self.nSW[self.barrierSW] = self.nNE[self.barrier]

    # Collide function
    def collide(self, omega, u0):

        rho = (
            self.n0
            + self.nN
            + self.nS
            + self.nE
            + self.nW
            + self.nNE
            + self.nSE
            + self.nNW
            + self.nSW
        )
        ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / rho
        uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / rho
        ux2 = ux * ux
        uy2 = uy * uy
        u2 = ux2 + uy2
        omu215 = 1 - 1.5 * u2
        uxuy = ux * uy
        self.n0 = (1 - omega) * self.n0 + omega * self.four9ths * rho * omu215
        self.nN = (1 - omega) * self.nN + omega * self.one9th * rho * (
            omu215 + 3 * uy + 4.5 * uy2
        )
        self.nS = (1 - omega) * self.nS + omega * self.one9th * rho * (
            omu215 - 3 * uy + 4.5 * uy2
        )
        self.nE = (1 - omega) * self.nE + omega * self.one9th * rho * (
            omu215 + 3 * ux + 4.5 * ux2
        )
        self.nW = (1 - omega) * self.nW + omega * self.one9th * rho * (
            omu215 - 3 * ux + 4.5 * ux2
        )
        self.nNE = (1 - omega) * self.nNE + omega * self.one36th * rho * (
            omu215 + 3 * (ux + uy) + 4.5 * (u2 + 2 * uxuy)
        )
        self.nNW = (1 - omega) * self.nNW + omega * self.one36th * rho * (
            omu215 + 3 * (-ux + uy) + 4.5 * (u2 - 2 * uxuy)
        )
        self.nSE = (1 - omega) * self.nSE + omega * self.one36th * rho * (
            omu215 + 3 * (ux - uy) + 4.5 * (u2 - 2 * uxuy)
        )
        self.nSW = (1 - omega) * self.nSW + omega * self.one36th * rho * (
            omu215 + 3 * (-ux - uy) + 4.5 * (u2 + 2 * uxuy)
        )

        # Force steady flow at the boundaries
        self.nE[:, 0] = self.one9th * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        self.nW[:, 0] = self.one9th * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        self.nNE[:, 0] = self.one36th * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        self.nSE[:, 0] = self.one36th * (1 + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        self.nNW[:, 0] = self.one36th * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        self.nSW[:, 0] = self.one36th * (1 - 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
        return ux, uy, rho

    # Calculate curl
    def curl(self, ux, uy):
        return (
            np.roll(uy, -1, axis=-1)
            - np.roll(uy, 1, axis=-1)
            - np.roll(ux, -1, axis=-2)
            + np.roll(ux, 1, axis=-2)
        )

    def initialize_arrays(self, u0):
        ones = np.ones((self.height, self.width), dtype=np.float32)
        n = {}
        # Initialize arrays
        u02 = u0**2
        self.n0 = self.four9ths * (ones - 1.5 * u02)
        self.nN = self.one9th * (ones - 1.5 * u02)
        self.nS = self.one9th * (ones - 1.5 * u02)
        self.nE = self.one9th * (ones + 3 * u0 + 3.0 * u02)
        self.nW = self.one9th * (ones - 3 * u0 + 3.0 * u02)
        self.nNE = self.one36th * (ones + 3 * u0 + 3.0 * u02)
        self.nSE = self.one36th * (ones + 3 * u0 + 3.0 * u02)
        self.nNW = self.one36th * (ones - 3 * u0 + 3.0 * u02)
        self.nSW = self.one36th * (ones - 3 * u0 + 3.0 * u02)
        rho = (
            self.n0
            + self.nN
            + self.nS
            + self.nE
            + self.nW
            + self.nNE
            + self.nSE
            + self.nNW
            + self.nSW
        )
        ux = (self.nE + self.nNE + self.nSE - self.nW - self.nNW - self.nSW) / rho
        uy = (self.nN + self.nNE + self.nNW - self.nS - self.nSE - self.nSW) / rho
        return n, rho, ux, uy

    def generate_random_barrier(self):
        # Clear the grid (start with no obstacles)
        barrier = np.zeros((self.height, self.width), bool)

        # Generate a random straight line
        length = np.random.normal(
            loc=10, scale=3
        )  # Random length (normal distribution)
        length = max(4, int(length))  # Ensure length is at least 4
        x = random.randint(int(self.width / 2) - 20, int(self.width / 2) - 10)
        y = random.randint(5, self.height - length - 5)
        barrier[y : y + length, x] = 1
        self.barrier = barrier

        self.barrierN = np.roll(self.barrier, 1, axis=0)
        self.barrierS = np.roll(self.barrier, -1, axis=0)
        self.barrierE = np.roll(self.barrier, 1, axis=1)
        self.barrierW = np.roll(self.barrier, -1, axis=1)
        self.barrierNE = np.roll(self.barrierN, 1, axis=1)
        self.barrierNW = np.roll(self.barrierN, -1, axis=1)
        self.barrierSE = np.roll(self.barrierS, 1, axis=1)
        self.barrierSW = np.roll(self.barrierS, -1, axis=1)
        return self.barrier

    def normalize(self, X, Y):
        # Calculate mean and std per channel (dim=0 excludes batch dimension)
        self.mean_X = X.mean(axis=(0, 2, 3), keepdims=True)  # Mean per channel
        self.std_X = X.std(axis=(0, 2, 3), keepdims=True)  # Std per channel

        self.mean_Y = Y.mean(axis=(0, 2, 3), keepdims=True)  # Mean per channel
        self.std_Y = Y.std(axis=(0, 2, 3), keepdims=True)  # Std per channel

        # Normalize by subtracting the mean and dividing by the std
        normalized_X = (X - self.mean_X) / self.std_X
        normalized_Y = (Y - self.mean_Y) / self.std_Y
        return normalized_X, normalized_Y


if __name__ == "__main__":
    gfd = GenerateFlowData((60, 120), (0.1, 0.1), (0.004, 0.005))

    x, y = gfd.gen_X_and_Y()

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    for i in range(len(y)):
        fluidImage = plt.imshow(
            y[i * 10],
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
