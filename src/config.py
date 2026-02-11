import yaml

def load_params(path="params.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
