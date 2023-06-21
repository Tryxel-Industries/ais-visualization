import json
import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def _parse_boosting_info(prop_values: dict, prop_type: str, enum_name: str):
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
        # v = np.add.reduceat(v, np.arange(0, len(v), 5))
        df_dict[f"{prop_type}_{k}"] = v

    return (prop_type, df_dict)

def _parse_single_v_prop(prop_values: dict, prop_type: str, boosting_rounds: int = 1):
    df_list = []
    for p in prop_values:
        if type(p) == str:
            df_list.append(pd.NA)
        else:
            # v = np.add.reduceat(v, np.arange(0, len(v), 5))
            df_list.append(p["SingleValue"])

    return (prop_type, {prop_type: df_list})
def parse_property(prop_dict: dict):
    prop_type = prop_dict["property_type"]
    prop_values = prop_dict["prop_values"]
    boosting_rounds = 1
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
            return _parse_boosting_info(prop_values, prop_type, "MappedInts")
        case "BoostAccuracyTest":
            # boosting_rounds = len(prop_dict["BoostAccuracyTest"]) / len(prop_dict["TestAccuracy"])
            return _parse_boosting_info(prop_values, prop_type, "MappedInts")
        case "ScoreComponents":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedFloats")
        case "AvgTrainScore":
            return _parse_single_v_prop(prop_values, prop_type)
        case "FoldAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "MappedFloats")
        case "Runtime":
            return _parse_single_v_prop(prop_values, prop_type, boosting_rounds=boosting_rounds)
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
    boosting_rounds = max(0, int(json_obj["params"]["boost"]))
    df_dict_list = []
    num_folds = 1
    for prop in meta_props:
        if prop["property_type"] == "FoldAccuracy":
            num_folds = len(prop["prop_values"])
        result = parse_property(prop)
        df_dict_list.append(result)


    df_dict_list.sort()
    df_dict = {}
    # num_folds = df_dict_list
    for (key, prop_dict) in df_dict_list:
        for k, v in prop_dict.items():
            if len(v) > num_folds:
                prop_dict[k] = np.add.reduceat(v, np.arange(0, len(v), boosting_rounds))[:num_folds]
        df_dict = {**df_dict, **prop_dict}

    return pd.DataFrame.from_dict(df_dict)

def get_some_sweet_meta_results(path: str, constraint: str = ""):
    print(os.getcwd())
    logs = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    meta_concise = []
    for log in logs:
        if constraint not in log:
            continue
        # full_path = path + "/" + log
        # print(full_path)
        with open(log, "r") as f:
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
            print(js_obj["params"]["antigen_pop_size"]["Fraction"])
            fold_rt_repl = js_obj["params"]["antigen_pop_size"]["Fraction"]
            meta_concise.append([js_obj["dataset"], fold_test_mean, fold_test_std, fold_train_mean, fold_train_std, fold_rt_mean, fold_rt_std, fold_rt_repl])
    df = pd.DataFrame(meta_concise, columns=["Dataset", "FoldAccuracy_test", "FoldAccuracy_test_std", "FoldAccuracy_train", "FoldAccuracy_train_std", "Runtime", "Runtime_std", "Pop_frac"])
    return df

def write_image(fig, path_str:str, filename:str):
    fig.update_layout(height = 500)
    path = Path(f"../plots/{path_str}")
    if not path.is_dir():
        path.mkdir(parents=True)
    fig.write_image(f"../plots/{path_str}/{filename}", scale=2)


def plot_agg_vals(df_list: List[pd.DataFrame], prefix: str, file_pre: str, sample_rate: int = 1, first_n = None) -> go.Figure:
    cols = df_list[0].columns
    fig = go.Figure()
    markers = ["circle", "square", "diamond", "cross", "star", "triangle-up", "triangle-down", "triangle-left", "triangle-right", "asterisk", "hash", "diamond-x"]
    n = 0

    col_type_map = {}
    for col in cols:
        # if col.startswith(prefix):
        if prefix in col:
            col_type_map[col] = []

    for df in df_list:
        for (k,v) in col_type_map.items():
            v.append(df[k])
            # a = df[k]
            # if a is not None:
            #     a  = a /a[0]
            #     v.append(a)

    for (k, v) in col_type_map.items():
        #agg_df = pd.DataFrame(v)
        agg_df = pd.concat(v,axis=1)
        #display(agg_df)
        # agg_df = agg_df[df.index % sample_rate == 0]
        if first_n == None:
                first_n = len(agg_df)
        agg_df = agg_df.iloc[:first_n:sample_rate, :]
        mean_vals = agg_df.mean(axis=1)
        std_vals = agg_df.std(axis=1)
        #display(mean_vals)

        fig.add_trace(go.Scatter(x=agg_df.index, y=mean_vals.ffill(),
                            mode='lines+markers',
                            name=k[len(prefix):],
                            marker=dict(
                                        symbol=markers[n],
                                        size=8,
                                        #angleref="previous",
                                    ),
                            error_y=dict(
                                     type='data',  # value of error bar given in data coordinates
                                     array=std_vals,
                                     visible=True)
                                 ))
        n += 1
    return fig
    fig.update_layout(height = 500)
    path = Path(f"../plots/{file_pre}/{prefix}")
    if not path.is_dir():
        path.mkdir(parents=True)
    # 
    # fig.update_xaxes(range=[1.5, 4.5])
    #fig.update_yaxes(range=[0.25, 0.405])
    fig.write_image(f"../plots/{file_pre}/{prefix}/agg.png", scale=2)
 
    fig.show()

def main():
    with open("./../ais/logg_dat/ex_boosting/logg_dat_Diabetes_boosting.json", "r") as f:
        js_obj = json.load(f)
    df_list = parse_iter_props(js_obj)
    df = parse_meta_props(js_obj)
    plot_agg_vals(df_list, "PopDimTypeMemberships_", "ajk")
    print(os.getcwd())
    get_some_sweet_meta_results("./../ais/logg_dat")
    exit()
    with open("data_in/test_dat.json", "r") as f:
        js_obj = json.load(f)


    print(js_obj)

if __name__ == '__main__':
    main()