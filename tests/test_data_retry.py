from types import MethodType

from qwen_vl.data import data_qwen
from qwen_vl.data.data_qwen import LazySupervisedDataset


def test_missing_sample_fallbacks_are_distinct_and_can_escape_a_bad_trajectory(
    monkeypatch,
):
    dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
    dataset.list_data_dict = [{} for _ in range(20)]
    attempted_indices = []

    def get_item(_self, index):
        attempted_indices.append(index)
        if index in {0, 1, 2, 3, 4}:
            raise FileNotFoundError(f"missing trajectory for {index}")
        return {"selected_index": index}

    dataset._get_item = MethodType(get_item, dataset)
    monkeypatch.setattr(data_qwen.time, "sleep", lambda _seconds: None)
    data_qwen.random.seed(42)

    result = dataset[0]
    assert result["selected_index"] not in {0, 1, 2, 3, 4}
    assert attempted_indices[:3] == [0, 0, 0]
    fallback_indices = attempted_indices[3:]
    assert len(fallback_indices) == len(set(fallback_indices))
    assert 0 not in fallback_indices
