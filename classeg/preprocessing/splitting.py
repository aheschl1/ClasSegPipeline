from typing import List, Dict

from sklearn.model_selection import train_test_split, KFold
from classeg.dataloading.datapoint import Datapoint


class Splitter:
    """
    For splitting data into folds.
    """

    def __init__(self, data: List[Datapoint], folds: int) -> None:
        assert folds > 0, 'Folds must be > 0.'
        self.data = data
        self.folds = folds

    def get_split_map(self) -> Dict[int, Dict[str, List[str]]]:
        results = {}
        if self.folds > 1:
            folder = KFold(n_splits=self.folds, shuffle=True)
        else:
            # Sklearn KFold only can handle folds > 1. Here, if we only have one fold, we just do a train/test split.
            for i in range(self.folds):
                xtrain, xtest = train_test_split(self.data, random_state=i, shuffle=True)
                results[i] = {
                    'train': [x.case_name for x in xtrain],
                    'val': [x.case_name for x in xtest]
                }
            return results
        # if we still exist, then we continue with the KFold
        for i, (train_idxs, test_idxs) in enumerate(folder.split(self.data)):
            results[i] = {
                'train': [self.data[x].case_name for x in train_idxs],
                'val': [self.data[x].case_name for x in test_idxs]
            }
        return results


class PatientSplitter:
    """
    For splitting data into folds.
    """

    def __init__(self, data: List[Datapoint], folds: int) -> None:
        assert folds > 0, 'Folds must be > 0.'
        self.data = data
        self.folds = folds

    def get_split_map(self) -> Dict[int, Dict[str, List[str]]]:
        results = {}
        if self.folds > 1:
            folder = KFold(n_splits=self.folds, shuffle=True)
        else:
            # Sklearn KFold only can handle folds > 1. Here, if we only have one fold, we just do a train/test split.
            for i in range(self.folds):
                xtrain, xtest = train_test_split(self.data, random_state=i, shuffle=True)
                results[i] = {
                    'train': [x.case_name for x in xtrain],
                    'val': [x.case_name for x in xtest]
                }
            return results
        # if we still exist, then we continue with the KFold
        for i, (train_idxs, test_idxs) in enumerate(folder.split(self.data)):
            results[i] = {
                'train': [self.data[x].case_name for x in train_idxs],
                'val': [self.data[x].case_name for x in test_idxs]
            }
        return results


if __name__ == "__main__":
    dummy_data = [Datapoint('faafas', 2, case_name='fds') for _ in range(10)]
    splitter = Splitter(dummy_data, 3)
    print(splitter.get_split_map())
