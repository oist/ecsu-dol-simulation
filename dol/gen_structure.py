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



DEFAULT_GEN_STRUCTURE = lambda d,n: load_genotype_structure('config/genotype_structure_{}d_{}n.json'.format(d,n))

if __name__ == "__main__":
    d=1
    n=2
    default_gs = DEFAULT_GEN_STRUCTURE(d,n)
    print("Size: {}".format(get_genotype_size(default_gs)))
    print("Neurons: {}".format(get_num_brain_neurons(default_gs)))
    print("DEFAULT_GEN_STRUCTURE: {}".format(json.dumps(default_gs, indent=3)))
