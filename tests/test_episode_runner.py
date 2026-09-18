import json
from qwen_vl.train.run_episode import final_checkpoint_path

def test_final_checkpoint_preserves_unlabeled_tail(tmp_path):
    periodic=tmp_path/'checkpoint-1'
    periodic.mkdir()
    (periodic/'manifest.json').write_text(json.dumps({'total_observations':64}))
    assert final_checkpoint_path(tmp_path,1,64)==periodic
    assert final_checkpoint_path(tmp_path,1,65)==tmp_path/'checkpoint-1-observations-65'
