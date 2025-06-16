import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from cup1d.utils.utils import get_path_repo


def in_hull(hull, p):
    return np.all(hull.eq @ p.T + hull.eq2[:, : p.shape[0]] <= hull.tol, 0)


class Hull(object):
    """
    A class for computing and working with the convex hull of a dataset, with optional scaling.

    This class computes the convex hull of a given dataset, optionally scaling the data before
    calculating the hull. The data is first centered by subtracting the mean of the dataset, then scaled
    by a specified factor (`extra_factor`). The convex hull is then computed on the transformed data.
    The class also provides a method to check if a point is inside the computed convex hull.

    Attributes:
    -----------
    hull : scipy.spatial.ConvexHull
        A `ConvexHull` object that contains the vertices, simplices, and other information about the convex hull
        of the scaled dataset.

    Methods:
    --------
    in_hull(point):
        Checks if a given point lies inside the computed convex hull.

    """

    def __init__(
        self,
        zs=None,
        data_hull=None,
        suite="mpg",
        save=False,
        extra_factor=1.0,
        mpg_version="Cabayol23",
        nyx_version="Jul2024",
        recompute=False,
        tol=1e-12,
        multi_dim=False,
    ):
        """
        Initializes the Hull object by computing the convex hull of a given dataset with an optional scaling factor.

        This method centers the provided dataset by subtracting its mean and then scales it by a specified factor
        (`extra_factor`). The convex hull of the scaled dataset is computed and stored as a `ConvexHull` object.
        The convex hull is stored as an attribute of the class, allowing for further operations such as checking
        if a point is inside the hull.

        Parameters:
        -----------
        data_hull : numpy.ndarray
            A 2D array of shape (n_samples, n_features) representing the dataset for which the convex hull is to be computed.
            Each row corresponds to a data point, and each column represents a feature (dimension).

        extra_factor : float, optional, default=1.05
            A scaling factor applied to the centered dataset before computing the convex hull.
            A value greater than 1.0 expands the dataset, while a value less than 1.0 contracts it.
            The default value is 1.05, slightly expanding the data.

        Returns:
        --------
        None
            This is the constructor of the `Hull` class, so it does not return any value. The resulting `ConvexHull` object
            is stored as an attribute `self.hull`.

        Notes:
        -----
        - The dataset is centered by subtracting the mean of the data along each feature (dimension).
        - The convex hull is computed using the scaled dataset, and the resulting `ConvexHull` object contains
          the vertices, simplices, and other details about the convex hull.
        """

        self.nz = len(zs)
        self.zs = zs
        self.tol = tol
        if suite == "mpg":
            self.params = [
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]
        elif suite == "nyx":
            self.params = [
                "Delta2_p",
                "n_p",
                "alpha_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            ]

        if multi_dim == True:
            self.hull = None
            if recompute == False:
                if suite == "mpg":
                    self.hull = self.load_hull(suite, mpg_version=mpg_version)
                elif suite == "nyx":
                    self.hull = self.load_hull(suite, nyx_version=nyx_version)

            if self.hull is None:
                self.hull = self.set_hull(data_hull, extra_factor=extra_factor)
                if save:
                    self.save_hull(
                        suite, mpg_version=mpg_version, nyx_version=nyx_version
                    )
            self.set_in_hull(zs)
        else:
            self.hulls = self.set_hulls(data_hull, extra_factor=extra_factor)

    def set_hulls(self, points, extra_factor=1.0):
        int_factor = extra_factor - 0.01

        hulls = []
        for jj0 in range(points.shape[1]):
            for jj1 in range(points.shape[1]):
                if jj1 >= jj0:
                    continue

                data_hull = points[:, [jj0, jj1]]

                mean = data_hull.mean(axis=0)
                int_data = int_factor * (data_hull - mean) + mean
                ext_data = extra_factor * (data_hull - mean) + mean
                hull = ConvexHull(int_data)
                hull.eq = hull.equations[:, :-1]
                hull.eq2 = np.repeat(
                    hull.equations[:, -1][None, :], data_hull.shape[0], axis=0
                ).T
                hull.tol = self.tol

                mask = in_hull(hull, ext_data) == False
                data_for_hull = ext_data[mask]

                hull_2d = ConvexHull(data_for_hull)
                hull_2d.eq = hull_2d.equations[:, :-1]
                hull_2d.eq2 = np.repeat(
                    hull_2d.equations[:, -1][None, :], self.nz, axis=0
                ).T
                hull_2d.tol = self.tol
                hull_2d.dim0 = jj0
                hull_2d.dim1 = jj1
                hulls.append(hull_2d)

        return hulls

    def in_hulls(self, p, zs=None):
        for jj in range(len(self.hulls)):
            res = in_hull(
                self.hulls[jj], p[:, [self.hulls[jj].dim0, self.hulls[jj].dim1]]
            )
            if res.all() == False:
                return False

        return True

    def set_hull(self, data_hull, extra_factor=1.050):
        int_factor = extra_factor - 1e-3
        mean = data_hull.mean(axis=0)
        int_data = int_factor * (data_hull - mean) + mean
        ext_data = extra_factor * (data_hull - mean) + mean
        hull = ConvexHull(int_data)

        data_for_hull = []
        for ii in range(ext_data.shape[0]):
            if self._in_hull(hull, ext_data[ii]) == False:
                data_for_hull.append(ext_data[ii])
        data_for_hull = np.vstack(data_for_hull)

        return ConvexHull(data_for_hull)

    def _in_hull(self, hull, point):
        """
        Check if a point is inside the convex hull.

        Parameters:
        -----------
        point : array-like
            The point to check, expected to be of shape (n_features,) where n_features is the number of features
            (dimensions) of the dataset.

        Returns:
        --------
        bool
            True if the point is inside the convex hull, False otherwise.

        Notes:
        -----
        This method uses the plane equations of the convex hull (derived from its faces) to determine if the point
        lies within the convex hull. The convex hull is considered to enclose all points whose projections
        onto the faces of the hull satisfy the inequality defined by the hull's equations.
        """
        return np.all(
            np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= 0
        )

    def save_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        if suite == "nyx":
            folder = os.environ["NYX_PATH"]
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        np.save(fname, vars(self.hull))

    def load_hull(self, suite, mpg_version="Cabayol23", nyx_version="Jul2024"):
        if suite == "nyx":
            folder = os.environ["NYX_PATH"]
            fname = os.path.join(folder, "hull_Nyx23_" + nyx_version + ".npy")
        elif suite == "mpg":
            folder = os.path.join(get_path_repo("cup1d"), "data", "hull")
            fname = os.path.join(folder, "hull_" + mpg_version + ".npy")

        if not os.path.exists(fname):
            return None

        vars_hull = np.load(fname, allow_pickle=True).item()

        # create a tiny hull to fill it with that stored in disk
        hull = ConvexHull(vars_hull["_points"][:50])
        for key in vars_hull.keys():
            setattr(hull, key, vars_hull[key])

        return hull

    def plot_hull(self, points, test_points=None):
        # Visualization: Project onto all 2D pairs of dimensions
        n_dimensions = points.shape[1]
        fig, axes = plt.subplots(
            n_dimensions,
            n_dimensions,
            figsize=(12, 12),
            constrained_layout=True,
        )

        for i in range(n_dimensions):
            for j in range(n_dimensions):
                if j > i:
                    axes[i, j].set_visible(False)
                    continue

                # Plot the points projected onto dimensions (i, j)
                if i == j:
                    axes[i, j].hist(points[:, i])
                else:
                    axes[i, j].scatter(points[:, j], points[:, i], s=10)

                    # uncomment for test points
                    # for icol in range(2):
                    #     col = "C" + str(icol + 2)
                    #     _ = np.argwhere(results == icol)[:, 0]
                    # axes[i, j].scatter(test_points[_, j], test_points[_, i], s=20, color=col)

                    # Project points onto dimensions (i, j)
                    projected_points = self.hull.points[:, [j, i]]
                    # Extract the hull vertices and sort them for the contour
                    projected_hull_points = projected_points[self.hull.vertices]
                    hull_2d = ConvexHull(projected_hull_points)
                    for simplex in hull_2d.simplices:
                        axes[i, j].plot(
                            projected_hull_points[simplex, 0],
                            projected_hull_points[simplex, 1],
                            "k-",
                        )

        for j in range(n_dimensions):
            axes[-1, j].set_xlabel(self.params[j])
            axes[j, 0].set_ylabel(self.params[j])

    def plot_hulls(self, points, test_points=None):
        # Visualization: Project onto all 2D pairs of dimensions
        n_dimensions = points.shape[1]
        fig, axes = plt.subplots(
            n_dimensions,
            n_dimensions,
            figsize=(12, 12),
            constrained_layout=True,
        )

        kk = 0
        for i in range(n_dimensions):
            for j in range(n_dimensions):
                if j > i:
                    axes[i, j].set_visible(False)
                    continue

                # Plot the points projected onto dimensions (i, j)
                if i == j:
                    axes[i, j].hist(points[:, i])
                else:
                    axes[i, j].scatter(points[:, j], points[:, i], s=10)

                    for simplex in self.hulls[kk].simplices:
                        axes[i, j].plot(
                            self.hulls[kk].points[simplex, 1],
                            self.hulls[kk].points[simplex, 0],
                            "k-",
                        )
                    kk += 1

        for j in range(n_dimensions):
            axes[-1, j].set_xlabel(self.params[j])
            axes[j, 0].set_ylabel(self.params[j])
