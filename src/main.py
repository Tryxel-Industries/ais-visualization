import json





class TestData:
    def __init__(self, dataset: str):
        self.dataset = dataset



def main():
    with open("data_in/test_dat.json", "r") as f:
        js_obj = json.load(f)
    print(js_obj)

if __name__ == '__main__':
    main()