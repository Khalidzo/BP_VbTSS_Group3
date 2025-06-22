import json
import pickle


def load_ground_truth_data(json_path, pkl_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    with open(pkl_path, "rb") as f:
        gt_data = pickle.load(f, encoding="latin1")

    gt_speeds = {car["carId"]: car["speed"] for car in gt_data["cars"]}
    for car in json_data["cars"]:
        car["real_speed"] = gt_speeds.get(car["id"])
    return json_data
