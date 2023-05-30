import json

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

    print(prop_type)
    match prop_type:
        case "PopDimTypeMemberships":
            return _parse_enum_dict_prop(prop_values, prop_type, "DimTypeMembership")

        case "PopLabelMemberships":
            return _parse_enum_dict_prop(prop_values, prop_type, "LabelMembership")

        case "TestAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "CorWrongNoReg")
        case "TrainAccuracy":
            return _parse_enum_dict_prop(prop_values, prop_type, "CorWrongNoReg")
        case "AvgTrainScore":
            return _parse_single_v_prop(prop_values, prop_type)
        case _:
            raise Exception("unconfigured prop type")


def gen_cor_w_noreg_cols(df: pd.DataFrame, key: str):
    n_cor = df[f"{key}_cor"]
    n_wrong = df[f"{key}_wrong"]
    n_no_reg = df[f"{key}_no_reg"]

    df[f"{key}_precision"] = n_cor/(n_cor+n_wrong)
    df[f"{key}_accuracy"] = n_cor/(n_cor+n_wrong+n_no_reg)


def parse_dataset(json_obj: dict) :
    dataset_name = json_obj["dataset"]

    iter_props = json_obj["iter_props"]

    df_dict_list = []
    for fold in iter_props:
        for prop in fold:
            df_dict_list.append(parse_property(prop))


    df_dict_list.sort()
    df_dict = {}
    for (key, prop_dict) in df_dict_list:
        df_dict = {**df_dict, **prop_dict}
    return pd.DataFrame.from_dict(df_dict)


def main():
    with open("data_in/test_dat.json", "r") as f:
        js_obj = json.load(f)


    print(js_obj)

if __name__ == '__main__':
    main()