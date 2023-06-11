import json
import os

import pandas as pd


class TestData:
    def __init__(self, dataset: str):
        self.dataset = dataset


def _parse_enum_dict_prop(prop_values: dict, prop_type: str, enum_name: str):
    cols = {}
    for p in prop_values:
        if type(p) == str:
            if len(cols) == 0:
                continue
            else:
                for k, v in cols.items():
                    v.append(pd.NA)
        else:
            memberships: dict = p[enum_name]
            for k, v in memberships.items():
                cur_val = cols.get(k)
                if cur_val is None:
                    cols[k] = [v]
                else:
                    cols[k].append(v)

    df_dict = {}
    for k, v in cols.items():
        df_dict[f"{prop_type}_{k}"] = v

    return (prop_type, df_dict)


def _parse_single_v_prop(prop_values: dict, prop_type: str):

    df_list = []
    for p in prop_values:
        if type(p) == str:
            df_list.append(pd.NA)
        else:
            df_list.append(p["SingleValue"])

    return (prop_type, {prop_type: df_list})
def parse_property(prop_dict: dict):
    prop_type = prop_dict["property_type"]
    prop_values = prop_dict["prop_values"]

    match prop_type:
        case "PopDimTypeMemberships":
            return _parse_enum_dict_prop(prop_values, prop_type, "DimTypeMembership")

        case "PopLabelMemberships":
            return _parse_enum_dict_prop(prop_values, prop_type, "LabelMembership")

        case "TestAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedInts")
        case "TrainAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedInts")
        case "BoostAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedInts")
        case "BoostAccuracyTest":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedInts")
        case "ScoreComponents":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedFloats")
        case "AvgTrainScore":
            return _parse_single_v_prop(prop_values, prop_type)
        case "FoldAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedFloats")
        case "Runtime":
            return _parse_single_v_prop(prop_values, prop_type)
        case _:
            raise Exception("unconfigured prop type")


def gen_cor_w_noreg_cols(df: pd.DataFrame, key: str):
    n_cor = df[f"{key}_cor"]
    n_wrong = df[f"{key}_wrong"]
    n_no_reg = df[f"{key}_no_reg"]

    df[f"{key}_precision"] = n_cor/(n_cor+n_wrong)
    df[f"{key}_accuracy"] = n_cor/(n_cor+n_wrong+n_no_reg)


def parse_iter_props(json_obj: dict):
    dataset_name = json_obj["dataset"]

    iter_props = json_obj["iter_props"]

    df_dict_list = []
    for fold in iter_props:
        fold_df_dict_list = []
        for prop in fold:
            fold_df_dict_list.append(parse_property(prop))
        df_dict_list.append(fold_df_dict_list)


    df_out_list = []
    for fold_df_list in df_dict_list:
        fold_df_list.sort()
        df_dict = {}
        for (key, prop_dict) in fold_df_list:
            df_dict = {**df_dict, **prop_dict}
        df_out_list.append(pd.DataFrame.from_dict(df_dict))
    return df_out_list

def parse_meta_props(json_obj: dict) :
    dataset_name = json_obj["dataset"]

    meta_props = json_obj["meta_props"]

    df_dict_list = []
    for prop in meta_props:
        df_dict_list.append(parse_property(prop))


    df_dict_list.sort()
    df_dict = {}
    for (key, prop_dict) in df_dict_list:
        df_dict = {**df_dict, **prop_dict}

    return pd.DataFrame.from_dict(df_dict)

def get_some_sweet_meta_results(path: str, constraint: str = ""):
    print(os.getcwd())
    logs = os.listdir(path)
    meta_concise = []
    for log in logs:
        if constraint not in log:
            continue
        full_path = path + "/" + log
        print(full_path)
        with open(full_path, "r") as f:
            js_obj = json.load(f)
            if not js_obj["meta_props"][0]["prop_values"]:
                continue
            loggy = parse_meta_props(js_obj)
            print(js_obj["meta_props"][0]["prop_values"])
            fold_test_std = loggy["FoldAccuracy_test"].std()
            fold_train_std = loggy["FoldAccuracy_train"].std()

            fold_test_mean = loggy["FoldAccuracy_test"].mean()
            fold_train_mean = loggy["FoldAccuracy_train"].mean()

            fold_rt_std = loggy["Runtime"].std()
            fold_rt_mean = loggy["Runtime"].mean()
            meta_concise.append([js_obj["dataset"], fold_test_mean, fold_test_std, fold_train_mean, fold_train_std, fold_rt_mean, fold_rt_std])
    df = pd.DataFrame(meta_concise, columns=["Dataset", "FoldAccuracy_test", "FoldAccuracy_test_std", "FoldAccuracy_train", "FoldAccuracy_train_std", "Runtime", "Runtime_std"])
    return df
def main():
    print(os.getcwd())
    get_some_sweet_meta_results("./../ais/logg_dat")
    exit()
    with open("data_in/test_dat.json", "r") as f:
        js_obj = json.load(f)


    print(js_obj)

if __name__ == '__main__':
    main()