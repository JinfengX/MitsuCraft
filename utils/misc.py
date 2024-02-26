# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
import re
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec

from plyfile import PlyElement, PlyData


def save_ply(file_path, vertices, colors=None):
    num_vertices = vertices.shape[0]

    vertex_properties = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    if colors is not None:
        vertex_properties += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_data = np.empty(num_vertices, dtype=vertex_properties)

    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]

    if colors is not None:
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    ply_data = PlyData([vertex_element])
    ply_data.write(file_path)

def insert_after_substring(original_str, target_substring, new_substring):
    """
    在原始字符串中的目标子字符串后插入新的子字符串
    :param original_str: 原始字符串
    :param target_substring: 目标子字符串
    :param new_substring: 要插入的新子字符串
    :return: 插入后的新字符串
    """
    index = original_str.find(target_substring)

    if index != -1:
        # 如果目标子字符串存在，则在目标子字符串后插入新的子字符串
        return (
            original_str[: index + len(target_substring)]
            + new_substring
            + original_str[index + len(target_substring) :]
        )
    else:
        # 如果目标子字符串不存在，则返回原始字符串
        return original_str


def extended_list(l, extend):
    """Update the list with another list.

    Args:
        l (list): The original list to be updated.
        extend (list): The list used to update the original list.

    Returns:
        list: The updated list.
    """
    l = l.copy()
    l.extend(extend)
    return l


def update_dict(d, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in d and isinstance(d[key], dict):
            update_dict(d[key], value)
        else:
            d[key] = value


def updated_dict(base_dict, updates):
    if updates is None:
        return base_dict
    updated = copy.deepcopy(base_dict)
    update_dict(updated, updates)
    return updated


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def is_dict(x):
    """Whether the input is an dict instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, dict)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def complie_regex(patterns):
    unique = set()
    for pt_str in patterns:
        if pt_str in unique:
            continue
        try:
            compiled_pt = re.compile(pt_str)
            unique.add(compiled_pt)
        except:
            warnings.warn(f"Invalid regex pattern: {pt_str}")
    return unique


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if args:
                arg_names = args_info.args[: len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead",
                            DeprecationWarning,
                        )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f"The expected behavior is to replace "
                            f"the deprecated key `{src_arg_name}` to "
                            f"new key `{dst_arg_name}`, but got them "
                            f"in the arguments at the same time, which "
                            f"is confusing. `{src_arg_name} will be "
                            f"deprecated in the future, please "
                            f"use `{dst_arg_name}` instead."
                        )

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead",
                            DeprecationWarning,
                        )
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper
