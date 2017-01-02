import unittest
import numpy as np
from validator.cv_methods import TrainTestSplitCV, TimeSeriesCV


class CVMethodsTestCase(unittest.TestCase):

    def test_train_test_split_cv(self):

        x = np.arange(20).reshape(10, 2)

        expected_trains = [[0, 1, 2, 3, 4, 5, 6, 7]]
        expected_tests = [[8, 9]]

        cv = TrainTestSplitCV(test_size=0.2)
        self.validate_cv(cv, x, expected_trains, expected_tests)

    def test_time_series_cv_fixed(self):

        window = 5
        horizon = 2

        x = np.arange(20).reshape(10, 2)

        """
        by = 1
        1) train = [0 1 2 3 4] test = [5 6]
        2) train = [1 2 3 4 5] test = [6 7]
        3) train = [2 3 4 5 6] test = [7 8]
        4) train = [3 4 5 6 7] test = [8 9]
        """

        expected_trains = [[0, 1, 2, 3, 4],
                           [1, 2, 3, 4, 5],
                           [2, 3, 4, 5, 6],
                           [3, 4, 5, 6, 7]]

        expected_tests = [[5, 6],
                          [6, 7],
                          [7, 8],
                          [8, 9]]

        cv = TimeSeriesCV(window, horizon, True)
        self.validate_cv(cv, x, expected_trains, expected_tests)

        """
        by = 2
        1) train = [0 1 2 3 4] test = [5 6]
        4) train = [3 4 5 6 7] test = [8 9]
        """

        cv = TimeSeriesCV(window, horizon, True, 2)
        self.validate_cv(cv, x, [expected_trains[0], expected_trains[3]], [expected_tests[0], expected_tests[3]])

    def test_time_series_cv_not_fixed(self):

        window = 5
        horizon = 2

        x = np.arange(20).reshape(10, 2)

        """
        by = 1
        1) train = [0 1 2 3 4] test = [5 6]
        2) train = [0 1 2 3 4 5] test = [6 7]
        3) train = [0 1 2 3 4 5 6] test = [7 8]
        4) train = [0 1 2 3 4 5 6 7] test = [8 9]
        """

        expected_trains = [[0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4, 5],
                           [0, 1, 2, 3, 4, 5, 6],
                           [0, 1, 2, 3, 4, 5, 6, 7]]

        expected_tests = [[5, 6],
                          [6, 7],
                          [7, 8],
                          [8, 9]]

        cv = TimeSeriesCV(window, horizon, False)
        self.validate_cv(cv, x, expected_trains, expected_tests)

        """
        by = 2
        1) train = [0 1 2 3 4] test = [5 6]
        4) train = [0 1 2 3 4 5 6 7] test = [8 9]
        """

        cv = TimeSeriesCV(window, horizon, False, 2)
        self.validate_cv(cv, x, [expected_trains[0], expected_trains[3]], [expected_tests[0], expected_tests[3]])

    def validate_cv(self, cv, x, expected_trains, expected_tests):
        actual_trains = []
        actual_tests = []

        for train, test in cv.split(x):
            actual_trains.append(list(train))
            actual_tests.append(list(test))

        self.assertListEqual(actual_trains, expected_trains)
        self.assertListEqual(actual_tests, expected_tests)


if __name__ == '__main__':
    unittest.main()
