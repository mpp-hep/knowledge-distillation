# Handles the data importing and small preprocessing for the interaction network.

import os
import numpy as np

from sklearn.model_selection import KFold

from .terminal_colors import tcols


class Data:
    """Data class to store the data to be used in learning for the interaction network.

    Attributes:
        data_folder: Path to the folder where the data is located.
        norm: How the data was normalised (nonorm if unnormalised).
        train_events: Number of events for the training sample, -1 to use all.
        test_events: Number of events for the testing sample, -1 to use all.
        pt_min: Minimum pt for any constituent in the data set.
        nconstituents: Number of constituents per jet.
        feature_selection: Type of feature selection scheme used. See the preprocessing
            module prepare_data.py for more details.
        jet_seed: Seed used in shuffling the jets.
        seed: The seed used in any shuffling that is done to the data.
    """

    def __init__(
        self,
        data_folder: str,
        norm: str = "nonorm",
        train_events: int = -1,
        test_events: int = -1,
        pt_min: str = "2",
        nconstituents: str = "128",
        feature_selection: str = "andre",
        jet_seed: int = None,
        seed: int = None,
    ):

        self.data_folder = data_folder
        self.norm_name = norm
        self.nconstituents = nconstituents
        self.minpt = pt_min
        self.feature_selection = feature_selection

        self.train_events = train_events
        self.test_events = test_events

        self.jet_seed = jet_seed
        self.seed = seed

        self.tr_data, self.tr_target = self._load_data("train")
        self.te_data, self.te_target = self._load_data("test")

        self.ntrain_jets = self.tr_data.shape[0]
        self.ntest_jets = self.te_data.shape[0]
        self.ncons = self.tr_data.shape[1]
        self.nfeat = self.tr_data.shape[2]

        self._success_message()

    @classmethod
    def shuffled(
        cls,
        data_folder: str,
        norm: str = "nonorm",
        train_events: int = -1,
        test_events: int = -1,
        pt_min: str = "2",
        nconstituents: str = "128",
        feature_selection: str = "andre",
        jet_seed: int = None,
        seed: int = None,
    ):
        """Shuffles the constituents. The jets are shuffled regardless."""
        data = cls(
            data_folder,
            norm,
            train_events,
            test_events,
            pt_min,
            nconstituents,
            feature_selection,
            jet_seed,
            seed,
        )

        print("Shuffling constituents...")
        rng = np.random.default_rng(seed)
        tr_seeds = rng.integers(low=0, high=10000, size=data.ntrain_jets)
        te_seeds = rng.integers(low=0, high=10000, size=data.ntest_jets)

        cls._shuffle_constituents(data.tr_data, tr_seeds, "training")
        cls._shuffle_constituents(data.te_data, te_seeds, "testing")

        return data

    @classmethod
    def _shuffle_constituents(cls, data: np.ndarray, seeds: np.ndarray, dtype: str):
        """Shuffle the constituents of a jet given an array of seeds equal in length
        to the number of jets in your data set.

        Args:
            data: Array containing the jet, constituents, and features.
            seeds: Array containing the seeds, equal in number to the jets.
            dtype: The type of data to shuffle, training or testing.
        """

        if data.shape[0] == 0:
            return data

        for jet_idx, seed in enumerate(seeds):
            shuffling = np.random.RandomState(seed=seed).permutation(data.shape[1])
            data[jet_idx, :] = data[jet_idx, shuffling]

        print(tcols.OKGREEN + f"Shuffled the {dtype} data! \U0001F0CF\n" + tcols.ENDC)

        return data

    def _load_data(self, data_type: str) -> tuple[np.ndarray, np.ndarray]:
        """Load data from the data files generated by the pre-processing scripts.

        Args:
            data_type: The type of data that you want to load, either train or test.

        Returns:
            Two numpy arrays with loaded data and the corresponding target.
        """
        datafile_name = self._get_data_filename(data_type)
        datafile_path = os.path.join(self.data_folder, datafile_name)

        targetfile_name = self._get_target_filename(data_type)
        targetfile_path = os.path.join(self.data_folder, targetfile_name)

        x = np.load(datafile_path)
        y = np.load(targetfile_path)

        nevents = self._get_nevents(data_type)
        x, y = self._trim_data(x, y, nevents, self.jet_seed)

        return x, y

    def _get_nevents(self, data_type: str) -> int:
        if data_type == "train":
            return self.train_events
        elif data_type == "test":
            return self.test_events
        else:
            raise TypeError(
                "Nevents error: type of data provided does not exist! "
                "Choose either 'train' or 'test'."
            )

    def _get_data_filename(self, data_type) -> str:
        return (
            "x_jet_images"
            + "_c"
            + self.nconstituents
            + "_pt"
            + self.minpt
            + "_"
            + self.feature_selection
            + "_"
            + self.norm_name
            + "_"
            + data_type
            + ".npy"
        )

    def _get_target_filename(self, data_type) -> str:
        return (
            "y_jet_images"
            + "_c"
            + self.nconstituents
            + "_pt"
            + self.minpt
            + "_"
            + self.feature_selection
            + "_"
            + self.norm_name
            + "_"
            + data_type
            + ".npy"
        )

    def _segregate_data(
        self, x_data: np.array, y_data: np.array
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Separates the data into separate arrays for each class.

        Args:
            x_data: Array containing the data to equalize.
            y_data: Corresponding onehot encoded target array.

        Returns:
            List of numpy arrays, each numpy array corresponding to a class of data.
            First list is for data and second list is corresponding target.
        """
        x_data_segregated = []
        y_data_segregated = []
        num_data_classes = y_data.shape[1]

        for data_class_nb in range(num_data_classes):
            class_elements_boolean = np.argmax(y_data, axis=1) == data_class_nb
            x_data_segregated.append(x_data[class_elements_boolean])
            y_data_segregated.append(y_data[class_elements_boolean])

        return x_data_segregated, y_data_segregated

    def _trim_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        maxdata: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cut the imported data and target and form new data sets with
        equal numbers of events per each class.

        Args:
            x: Numpy array containing the data.
            y: Numpy array containing the corresponding target (one-hot).
            maxdata: Maximum number of jets to load.
            seed: Seed to used in shuffling the jets after trimming.

        Returns:
            Two numpy arrays, one with data and one with target,
            containing an equal number of events per each class.
        """
        if maxdata < 0:
            return x, y

        num_classes = y.shape[1]
        maxdata_class = int(int(maxdata) / num_classes)

        x_segregated, y_segregated = self._segregate_data(x, y)

        x = x_segregated[0][:maxdata_class, :, :]
        y = y_segregated[0][:maxdata_class, :]

        for x_class, y_class in zip(x_segregated[1:], y_segregated[1:]):
            x = np.concatenate((x, x_class[:maxdata_class, :, :]), axis=0)
            y = np.concatenate((y, y_class[:maxdata_class, :]), axis=0)

        shuffling = np.random.RandomState(seed=seed).permutation(x.shape[0])
        return x[shuffling], y[shuffling]

    def _success_message(self):
        # Display success message for loading data when called.
        print("\n----------------")
        print(tcols.OKGREEN + "Data loading complete:" + tcols.ENDC)
        print(f"Training data size: {self.tr_data.shape[0]:.2e}")
        print(f"Test data size: {self.te_data.shape[0]:.2e}")
        print("----------------\n")
