def fetch_key_from_dictionary(dictionary, key, error_message):
    """
    Fetches a value from a dictionary
    :param dictionary: The dictionary to search
    :param key: The key to look-up
    :param error_message: The error message to print when the key is not found and an exception is raised.
    :return: The value corresponding to the key
    """

    try:
        return dictionary[key]
    except KeyError:
        raise KeyError(error_message)
