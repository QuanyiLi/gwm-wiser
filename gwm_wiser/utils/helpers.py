import copy
import datetime
import random

import torch


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into a deep copy of dict1.
    """
    # Create a deep copy to avoid modifying the original dict1
    merged = copy.deepcopy(dict1)

    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # If key exists in both and both values are dicts, recurse
            merged[key] = deep_merge(merged[key], value)
        else:
            # Otherwise, just set/overwrite the value
            # Use deepcopy for new values to maintain integrity
            merged[key] = copy.deepcopy(value)

    return merged


def get_max_depth(item):
    """Helper to find the maximum nesting depth of a list."""
    if not isinstance(item, list) or not item:
        return 0
    return 1 + max(get_max_depth(i) for i in item)


def repeat_interleave_at_depth(data, repeats, target_depth, current_depth=0):
    """
    Repeats elements only at a specific depth level.
    """
    # Base case: if we reached the target depth, repeat the entire sub-item
    if current_depth == target_depth:
        return [data] * repeats

    # If it's a list but not at the target depth, recurse
    if isinstance(data, list):
        new_list = []
        for item in data:
            result = repeat_interleave_at_depth(
                item, repeats, target_depth, current_depth + 1
            )
            # If the recursion returned a list of repetitions, flatten them into this level
            if current_depth + 1 == target_depth:
                new_list.extend(result)
            else:
                new_list.append(result)
        return new_list

    # If it's a leaf but we haven't reached depth, return as is
    return data


def repeat_at_level(data, repeats, level):
    """Wrapper to handle negative indexing for levels."""
    max_d = get_max_depth(data)
    # Convert negative level to positive (e.g., -1 becomes max_depth)
    actual_target = level if level >= 0 else max_d + level + 1
    return repeat_interleave_at_depth(data, repeats, actual_target)


def repeat_to_length(xs, M):
    if M <= 0:
        return []
    n = len(xs)
    if n == 0:
        return []
    # repeat enough times then slice
    reps = -(-M // n)  # ceil(M / n)
    first_column = xs * reps
    all_cubes_params = []
    for i in range(len(xs)):
        joint = first_column[i:] + xs[:i]
        all_cubes_params.append(joint[:M])
    return all_cubes_params


def generate_color(family, random_range=0):
    """
    Generates a random RGB color from a specified color family.

    Args:
        family (str): The desired color family.
                      Accepts "red", "green", or "blue".
                      Defaults to "red".
    """
    if family.lower() == "red":
        # Emphasize red, with some green and blue
        r = random.randint(255 - random_range, 255)
        g = random.randint(0, random_range)
        b = random.randint(0, random_range)
    elif family.lower() == "green":
        # Emphasize green, with some red and blue
        r = random.randint(0, random_range)
        g = random.randint(255 - random_range, 255)
        b = random.randint(0, random_range)
    elif family.lower() == "blue":
        # Emphasize blue, with some red and green
        r = random.randint(0, random_range)
        g = random.randint(0, random_range)
        b = random.randint(255 - random_range, 255)
    elif family.lower() == "yellow":
        r = random.randint(255 - random_range, 255)
        g = random.randint(255 - random_range, 255)
        b = random.randint(0, random_range)
    elif family.lower() == "purple":
        # Emphasize red and blue, with minimal green
        r = random.randint(180 - random_range, 180 + random_range)
        g = random.randint(0, random_range)
        b = random.randint(255 - random_range, 255)
    elif family.lower() == "orange":
        # High red, medium green, low blue
        r = random.randint(255 - random_range, 255)
        g = random.randint(140 - random_range, 140 + random_range)
        b = random.randint(0, random_range)
    elif family.lower() == "grey":
        # High red and blue, medium green
        r = random.randint(40 - random_range, 40)
        g = random.randint(40 - random_range, 40)
        b = random.randint(40 - random_range, 40)
    elif family.lower() == "cyan":
        # Low red, high green and blue
        r = random.randint(0, random_range)
        g = random.randint(255 - random_range, 255)
        b = random.randint(255 - random_range, 255)
    else:
        raise ValueError("Unknown color family '%s'" % family)

    return [r / 255, g / 255, b / 255, 1]


def batch_string_to_tensor(task_instruction):
    byte_lists = [list(t.encode("utf-8")) for t in task_instruction]
    max_len = max(len(b) for b in byte_lists)
    padded = [b + [0] * (max_len - len(b)) for b in byte_lists]
    t = torch.tensor(padded, dtype=torch.uint8)
    return t


def batch_tensor_to_string(tensor):
    strings = []
    for byte_tensor in tensor:
        byte_list = byte_tensor.tolist()
        # Remove padding zeros
        byte_list = [b for b in byte_list if b != 0]
        s = bytes(byte_list).decode("utf-8")
        strings.append(s)
    return strings


def current_time_str():
    return datetime.datetime.now().strftime("%m%d-%H%M%S")
