import os, sys, importlib, pickle, gzip
os.system("source /scratch/jve4pt/platiglib/platiglib_venv/bin/activate")

os.chdir("/scratch/jve4pt/platiglib/")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

platiglib = importlib.import_module("platiglib")
from platiglib.model.evaluation import ModelEvaluatorFactory
from platiglib.params import ParameterSetFactory
from platiglib.utils import load_config

cell_lines = ["K562", "HepG2"]


for cell_line in cell_lines:
    
    unique_ids = {}

    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               f"1_{cell_line}_no_sample_importance_downsampling.yaml")
    config_overlay = load_config(config_path)    
    param_set = ParameterSetFactory.new_from_overlay(platiglib_project="rbpse", dataset_name="RBPSEDataset",
                                                 model_name="XGBRegressor", overlay_dict=config_overlay)
    
    evaluator = ModelEvaluatorFactory.new_from_param_set(param_set)
    metadata_df = evaluator.ds.get_metadata()

    indices = {
        "train": evaluator.train_loader.dataset.indices,
        "val": evaluator.val_loader.dataset.indices,
        "test": evaluator.test_loader.dataset.indices
    }

    for key in indices.keys():
        tmp_indices = metadata_df.iloc[indices[key]]["unique_id"].tolist()
        assert len(tmp_indices) == len(set(tmp_indices)), f"Duplicate unique_ids found in {key} set for {cell_line}"
        unique_ids[key] = set(tmp_indices)

    # ensure pairwise disjointness of train/val/test unique_ids
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = unique_ids[a] & unique_ids[b]
        assert not inter, f"Found {len(inter)} overlapping unique_ids between {a} and {b} for {cell_line}. Examples: {list(inter)[:5]}"

    unique_ids["feature_order"] = evaluator.ds.get_flattened_feature_index()
    print(unique_ids["feature_order"].tolist())
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_dir, f"2_{cell_line}_unique_ids.pkl.gz")
    
    with gzip.open(out_path, "wb") as f:
        pickle.dump(unique_ids, f)

