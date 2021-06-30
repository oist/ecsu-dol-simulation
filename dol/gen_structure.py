"""
Reads genotype structure from a json file and validates it.
"""

import json


def get_num_brain_neurons(genotype_structure):
    """
    TODO: Missing function docstring
    """
    neural_gains = genotype_structure["neural_gains"]
    return len(neural_gains['indexes']) \
        if 'indexes' in neural_gains \
        else len(neural_gains['default'])


def get_genotype_size(genotype_structure):
    """
    TODO: Missing function docstring
    """
    return 1 + max(x for v in genotype_structure.values() \
                   if 'indexes' in v for x in v['indexes'])  # last index


def check_genotype_structure(genotype_structure):
    """
    Check consistency of genotype structure
    """
    num_genes = 1 + max(x for v in genotype_structure.values() \
                        if 'indexes' in v for x in v['indexes'])  # last index
    all_indexes_set = set(x for v in genotype_structure.values() \
                          if 'indexes' in v for x in v['indexes'])
    assert len(all_indexes_set) == num_genes
    for k, v in genotype_structure.items():
        if k == 'crossover_points':
            continue
        k_split = k.split('_')
        assert len(k_split) == 2
        assert k_split[0] in ['sensor', 'neural', 'motor']
        assert k_split[1] in ['taus', 'biases', 'gains', 'weights']
        assert k_split[0] == 'neural' or k_split[1] != 'taus'
        if 'indexes' in v:
            assert sorted(set(v['indexes'])) == sorted(v['indexes'])
        else:
            assert 'default' in v
            assert type(v['default']) == list
            assert type(v['default'][0]) == float
        # only neural have taus (sensor and motor don't)

    # check if all values in sensor* and motor* have the same number of indexes/default values
    set(
        len(v['indexes']) if 'indexes' in v else len(v['default'])
        for k, v in genotype_structure.items()
        if any(k.startswith(prefix) for prefix in ['sensor', 'motor'])
    )


def load_genotype_structure(json_filepath, process=True):
    """
    TODO: Missing function docstring
    """
    with open(json_filepath) as f_in:
        genotype_structure = json.load(f_in)

    if process:
        check_genotype_structure(genotype_structure)
    return genotype_structure

def build_structure(
        num_sensors = 2,
        num_neurons = 2,
        num_motors = 2,
        sensor_gains_range = [1,20],
        sensor_biases_range = [-3, 3],
        sensor_weights_range = [-8, 8],
        neural_taus_range = [1, 2],
        neural_biases_range = [-3, 3],
        neural_weights_range = [-8, 8],
        motor_gains_range = [1, 20],
        motor_biases_range = [-3, 3],
        motor_weights_range = [-8, 8]
    ):
    ci = 0 # current index
    structure = {}
    structure["sensor_gains"] = {
        "indexes": [ci],
        "range": sensor_gains_range
    }
    ci += 1
    structure["sensor_biases"] = {
        "indexes": [ci],
        "range": sensor_biases_range
    }
    ci += 1    
    num_sensor_weights = num_sensors * num_neurons
    structure["sensor_weights"] = {
        "indexes": list(range(ci, ci+num_sensor_weights)),
        "range": sensor_weights_range
    }
    ci += num_sensor_weights
    structure["neural_taus"] = {
        "indexes": [ci],
        "range": neural_taus_range
    }
    ci += 1   
    structure["neural_biases"] = {
        "indexes": [ci],
        "range": neural_biases_range
    }
    ci += 1
    structure["neural_gains"] = {
        "default": [1.0] * num_neurons
    }
    num_neural_weights = num_neurons * num_neurons
    structure["neural_weights"] = {
        "indexes": list(range(ci, ci+num_neural_weights)),
        "range": neural_weights_range
    }
    ci += num_neural_weights
    structure["motor_gains"] = {
        "indexes": [ci],
        "range": motor_gains_range
    }
    ci += 1
    structure["motor_biases"] = {
        "indexes": [ci],
        "range": motor_biases_range
    }
    ci += 1
    num_motor_weights = num_neurons * num_motors
    structure["motor_weights"] = {
        "indexes": list(range(ci, ci+num_motor_weights)),
        "range": motor_weights_range
    }
    return structure


DEFAULT_GEN_STRUCTURE_FROM_FILE = lambda d,n: load_genotype_structure('config/genotype_structure_{}d_{}n.json'.format(d,n))

DEFAULT_GEN_STRUCTURE = lambda d,n: build_structure(
    num_sensors = 2 * d,
    num_neurons = n,
    num_motors = 2 * d
)

if __name__ == "__main__":
    for d,n in [(1,2),(1,3),(1,4),(2,2),(2,3),(2,4)]:
        default_gs_file = DEFAULT_GEN_STRUCTURE_FROM_FILE(d,n)
        default_gs = DEFAULT_GEN_STRUCTURE(d,n)
        assert default_gs_file == default_gs
        # check_genotype_structure(default_gs)
        # print("Size: {}".format(get_genotype_size(default_gs)))
        # print("Neurons: {}".format(get_num_brain_neurons(default_gs)))
        # print("DEFAULT_GEN_STRUCTURE: {}".format(json.dumps(default_gs, indent=3)))
