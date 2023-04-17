


def dict_concatenation(dicts):
    """convert iterable of dictinnary to dictionnary of iterable, whose keys are the combination
    of all the possible keys, and whose values are the concatenation of all the values of 'dicts', with
    same key, such that absent values are filled with None

    Args:
        dicts (Iterable[dict])

    Raises:
        AssertionError: check if the dicts is an iterable of dicts, otherwise return error

    Returns:
        _type_: 
    """
    all_keys = set()
    for curr_dict in  dicts:
        assert isinstance(curr_dict,dict)
        all_keys.union(curr_dict.all_keys())

    output_dict = {key:[] for key in all_keys}

    for curr_dict in  dicts:
        for key in all_keys:
            if key in curr_dict:
                output_dict[key].append(curr_dict[key])
            else:
                output_dict[key].append(None)

    return output_dict
