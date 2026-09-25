import numpy as np
import pandas as pd
import warnings
import SRConfig as SRConfig

warnings.filterwarnings("ignore")


class SRData:
    x_data_all = None
    y_data_all = None
    x_data = None
    y_data = None
    x_val = None
    y_val = None
    nVal = None
    x_test = None
    y_test = None
    x_plot = None
    y_plot = None
    miniBatches: list[tuple] = None

    forwardpass_data = None
    forwardFullMask = None
    forwardpass_data_boundaries = {}
    forwardpass_masks = {}
    forwardpass_counts = {}

    def read_data_from_file(
        train_file: str = None, val_file: str = None, test_file=None
    ):
        """
        Reads data from "--train_data" file and splits them into training and validation sets according to the ratio specified by "--train_val" parameter.
        Reads test data from "--test_data" file.
        """

        def _load_file_data(filename):
            try:
                df = pd.read_csv(filename, header=None, skiprows=0)
                if df.shape[1] < 2:
                    raise Exception("Invalid data format")
            except Exception:
                df = pd.read_csv(filename, header=None, skiprows=2, sep=r"\s+")
            return df.to_numpy().reshape(df.shape)

        # -----------------------------------
        # --- Training and validation data
        # -----------------------------------
        data = _load_file_data(train_file)
        cols = list(range(data.shape[1] - 1))
        SRData.x_data_all = data[:, cols].reshape((data.shape[0], len(cols)))
        SRData.y_data_all = SRConfig.scaleCoeff * data[:, -1].reshape(
            (data.shape[0], 1)
        )
        # -------------------------------
        # --- Validation data provided
        # -------------------------------
        if val_file is not None:
            SRData.x_data = data[:, cols].reshape((data.shape[0], len(cols)))
            SRData.y_data = SRConfig.scaleCoeff * data[:, -1].reshape(
                (data.shape[0], 1)
            )
            valid_data = _load_file_data(val_file)
            SRData.x_val = valid_data[:, cols].reshape((valid_data.shape[0], len(cols)))
            SRData.y_val = SRConfig.scaleCoeff * valid_data[:, -1].reshape(
                (valid_data.shape[0], 1)
            )

        # -------------------------------
        # --- Validation data not provided
        # -------------------------------
        else:
            # --- Too few data samples
            if data.shape[0] <= 20:
                SRData.x_data = data[:, cols].reshape((data.shape[0], len(cols)))
                SRData.y_data = SRConfig.scaleCoeff * data[:, -1].reshape(
                    (data.shape[0], 1)
                )
                SRData.x_val = SRData.x_data
                SRData.y_val = SRData.y_data
            # --- Enough data samples
            else:
                # --- Shuffle training data
                data_ids = np.array(range(data.shape[0]))
                r_static = np.random.default_rng(SRConfig.seed)
                r_static.shuffle(data_ids)
                data = data[data_ids, :]
                # --- Training data
                firstN = int(SRConfig.train_val_split * data.shape[0])
                SRData.x_data = data[:firstN, cols].reshape((firstN, len(cols)))
                SRData.y_data = SRConfig.scaleCoeff * data[:firstN, -1].reshape(
                    (firstN, 1)
                )
                # -------------------------------
                # --- Validation data: ORIGINAL
                # -------------------------------
                lastN = data.shape[0] - firstN
                if lastN > 0:
                    SRData.x_val = data[data.shape[0] - lastN :, cols].reshape(
                        lastN, len(cols)
                    )
                    SRData.y_val = SRConfig.scaleCoeff * data[
                        data.shape[0] - lastN :, data.shape[1] - 1
                    ].reshape((lastN, 1))
                else:
                    SRData.x_val = data[:, cols].reshape(data.shape[0], len(cols))
                    SRData.y_val = SRConfig.scaleCoeff * data[
                        :, data.shape[1] - 1
                    ].reshape((data.shape[0], 1))

        SRData.nVal = SRData.x_val.shape[0]
        # ----------------
        # --- Test data
        # ----------------
        if test_file is not None:
            df = pd.read_csv(test_file, header=None, skiprows=0)
            data = df.to_numpy().reshape(df.shape)
            SRData.x_test = data[:, cols].reshape((data.shape[0], len(cols)))
            SRData.y_test = SRConfig.scaleCoeff * data[:, data.shape[1] - 1].reshape(
                (data.shape[0], 1)
            )
        else:
            SRData.x_test = SRData.x_data
            SRData.y_test = SRData.y_data

    @staticmethod
    def target_scale_std() -> float:
        """
        Robust magnitude scale of the (scaled) training targets, used to make
        pruning thresholds data-relative instead of hard-wired absolute values.
        Returns std(y_data) (the spread a hidden unit must help reproduce), or
        0.0 if there is no data / the value is non-finite. Recomputed on demand;
        y_data is refreshed on every data reload, so the scale tracks the
        current dataset version automatically.
        """
        y = SRData.y_data
        if y is None or len(y) == 0:
            return 0.0
        s = float(np.std(y))
        return s if np.isfinite(s) else 0.0

    def init_data(train_file: str = None, val_file: str = None, test_file=None):
        SRData.read_data_from_file(
            train_file=train_file, val_file=val_file, test_file=test_file
        )
        # log(
        #     f"training data: {SRData.x_data.shape[0]}, \nvalidation data: {SRData.x_val.shape[0]}, \ntest data: {SRData.x_test.shape[0]}",
        #     SRConfig.start_time,
        #     level="HIGHLIGHT_MAGENTA",
        # )

        # --- Training data
        SRData.forwardpass_data = SRData.x_data
        SRData.forwardpass_data_boundaries["train"] = (0, SRData.x_data.shape[0] - 1)
        SRData.forwardpass_counts["train"] = SRData.x_data.shape[0]
        # --- Validation data
        SRData.forwardpass_data_boundaries["valid"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_val.shape[0] - 1,
        )
        SRData.forwardpass_counts["valid"] = SRData.x_val.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_val, axis=0
        )
        # --- Test data
        SRData.forwardpass_data_boundaries["test"] = (
            SRData.forwardpass_data.shape[0],
            SRData.forwardpass_data.shape[0] + SRData.x_test.shape[0] - 1,
        )
        SRData.forwardpass_counts["test"] = SRData.x_test.shape[0]
        SRData.forwardpass_data = np.append(
            SRData.forwardpass_data, SRData.x_test, axis=0
        )

        return SRData
