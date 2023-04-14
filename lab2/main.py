import json
from reader.data_reader import DataReader
from reader.csv_reader import CsvReader
from coder.oil_coder import OilCoder
from coder.mushrooms_coder import MushroomsCoder

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    data_file_name = config['data_file']
    is_oil = data_file_name.endswith('.csv')
    reader = CsvReader(data_file_name) if is_oil else DataReader(data_file_name)
    str_attr = reader.get_values(config['start_row'],
                                 config["end_row"],
                                 config['start_attributes'],
                                 config['end_attributes'])
    str_targets = reader.get_values(config['start_row'],
                                    config["end_row"],
                                    config['start_targets'],
                                    config['end_targets'])
    str_attr = {sample_id : attr_sample for sample_id, attr_sample in
                str_attr.items() if sample_id in str_targets}
    coder = OilCoder() if is_oil else MushroomsCoder()
    attributes = coder.encode(str_attr.values(), withNones=False)
    targets = coder.encode(str_targets.values(), withNones=True)
    attr_deltas, attr_mins, norm_attr = coder.normalize(list(
        attributes.values()))
    target_deltas, target_mins, norm_targets = coder.normalize_targets(
        list(targets.values()))

    normalized_attrs, normalized_targets = coder.get_normalized_samples(list(
        str_targets.keys()), norm_attr, norm_targets)
