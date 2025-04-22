def _binary_encode(data, column, out_values=[-1, 1], reverse=False):
    unique_values = data[column].unique()
    if reverse:
        # Flip unique_values order
        unique_values = unique_values[::-1]
    if len(unique_values) != 2:
        raise ValueError(f"{column} must contain exactly two unique values.")
    print(
        "Recoding '", unique_values[0], "' as ", -1,
        " and '", unique_values[1], "' as ", 1, sep=''
    )
    value_map = {
       unique_values[0]: out_values[0], unique_values[1]: out_values[1]
    }
    data[column] = data[column].map(value_map)
    return data


def _is_binary_neg_pos(col):
    """
    Check if a column in a DataFrame is binary.
    A column is considered binary if it contains exactly two unique values.
    """
    try:
        return set(col.str[:3].unique()) in [{'Neg', 'Pos'}]
    except:
        return False
