from typing import Any, List, Union

import numpy as np
from modules.utils.common import stable_sort


class NumerizeLabels:
    """
    Converts a list of string labels to numeric labels. 
    Use stable sort by default i.e. ['1', '10', '2'] would be sorted as ['1', '2', '10']
    """

    def __init__(self, uniq_labels_list: Union[List[Any], np.ndarray], use_stable_sort: bool = True) -> None:
        if use_stable_sort:
            uniq_labels_list = stable_sort(uniq_labels_list)
        self.numeric_label_dict = {
            label: i for i,
            label in enumerate(uniq_labels_list)}

    def __call__(self, label: Any):
        return self.numeric_label_dict[label]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        return format_string
