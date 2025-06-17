import pandas


def _binary_encode(data, column, out_values=[-1, 1], reverse=False):
    if isinstance(column, list):
        reference = column[1]
        column = column[0]
        # Replace all instances of data[column] that are not equal to reference
        # with "not reference"
        data[column] = [
            reference if row == reference else f"not {reference}"
            for row in data[column]
        ]

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
    try:
        return set(col.str[:3].unique()) in [{'Neg', 'Pos'}]
    except Exception:
        return False


def _auto_binary_encode(data):
    # Sweep all columns. If coded as {Neg*, Pos*}, recode to {-1, 1}
    for col in data.columns:
        if _is_binary_neg_pos(data[col]):
            data = _binary_encode(data, col)
    return data


def _auto_get_dummies(data, min_levels=3, max_levels=5):
    cat_vars = data.select_dtypes('object').columns  # Start with all cols

    # 'Response' doesn't need recoding
    cat_vars = cat_vars.drop('Response', errors='ignore')

    for col in cat_vars:
        # Drop columns with too many or too few unique values
        n_unique = len(data[col].unique())
        if n_unique > max_levels or n_unique < min_levels:
            cat_vars = cat_vars.drop(col)

    # If no categorical variables are found, set cat_vars to None
    if len(cat_vars) > 0:
        data = pandas.get_dummies(data, columns=cat_vars)

    return data
