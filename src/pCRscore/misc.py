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

import pandas

def _auto_get_dummies(data, min_levels=3, max_levels=5):
    """
    Check if a column in a DataFrame is multilevel.
    A column is considered multilevel if it contains more than min_levels unique values.
    """
    cat_vars = data.select_dtypes('object').columns # Start with all cols

    for col in cat_vars:
        # Drop columns with too many or too few unique values
        n_unique = len(data[col].unique())
        if n_unique > max_levels or n_unique < min_levels:
            cat_vars = cat_vars.drop(col)

    # Remove the 'Response' column from cat_vars
    cat_vars = [col for col in cat_vars if col != 'Response']

    # If no categorical variables are found, set cat_vars to None
    if len(cat_vars) > 0:
        data = pandas.get_dummies(data, columns=cat_vars)

    return data
