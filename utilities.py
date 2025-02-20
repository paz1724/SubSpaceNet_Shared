import sys
import numpy as np
import platform
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path


curr_os = platform.system()


def get_path_config(for_eval=False):
    """ this function return a dictionary of the main configuration file"""
    pathStr = str(Path(__file__).parent)
    json_path = translate_path(pathStr)
    json_path = os.path.join(json_path, 'config.json') if not for_eval else (
        os.path.join(json_path, 'evaluation_config.json'))
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    data = dict(data)
    for key in data.keys():
        if (type(key) == str) & ('path' in key):
            data[key] = translate_path(data[key])
    return data


def translate_path(in_path):
    """ use it for any path when using cnvrg
    example: my_path = translate_path(any_path)"""
    computer_os_name = sys.platform
    if computer_os_name == 'linux':
        if in_path[0:4] != '/mnt':
            in_path = convert_path_to_linux(in_path)
        return in_path
    elif computer_os_name == 'win32':
        return (in_path)
    else:
        raise Exception('Computer Os Name %s not supported. Only win32, linux supported.')
    # else windows, do nothing


def convert_path_to_linux(path):
    """
    help function for translate path
    """
    if path is None:
        return path
    if curr_os != 'Windows':
        if path[0:4] == '/mnt':   # Already linux format
            return path

        if path[0:2] == 'Z:' or path[0:2] == 'z:':
            path = r'\\netapp6a\vered_hkb' + path[2::]

        if path[0:2] == 'Q:' or path[0:2] == 'q:':
            path = r'\\netapp1\vol2\home4\ir' + path[2::]

        if path[0:2] == 'T:' or path[0:2] == 't:':
            path = r'\\netapp3\cnvrg_rw' + path[2::]

        if path[0:2] == 'N:' or path[0:2] == 'n:':
            path = r'\\netapp1\hkcommon' + path[2::]

        if r'netapp6\\' in path:
            path = path.replace(r'netapp6\\', r'netapp6a\\')

        if '/' in path or '\\' in path:
            # Convert tail_file_path to linux
            path = path.replace('\\', '/')
            path = r'/mnt' + path[1::]

    return path


def _serialize(**kwargs):
    """ help function for creating the serialization in procedural way"""
    str_to_return = ""
    for key, value in kwargs.items():
        str_to_return += f"{key}_{value}_"
    return str_to_return[:-1]


def serialize(weights_path, dataset_path):
    """ returns full serialization according to weights and dataset"""
    weights_snr = weights_path.split("SNR=")[1].split("_")[0]
    diff_method = weights_path.split('diff_method=')[1].split("_")[0]
    weights_version = get_version(weights_path)
    if 'M=rand' not in weights_path:
        number_of_sources_for_train = weights_path.split("M=")[1].split("_")[0]
    snapshots = get_snapshots(weights_path)
    # test same snapshots in trained weights and in datasets
    snapshots_test = dataset_path.split('T=')[1].split("_")[0]
    assert snapshots_test == snapshots
    # dataset of evaluation-relevant serialization:

    array_form = get_array_form(dataset_path)
    number_of_sources = dataset_path.split('M=')[1].split('_')[0]
    examples_in_eval_data = dataset_path.split('non-coherent_')[1].split('_')[0]
    snr_eval_dataset = dataset_path.split('SNR=')[1].split('_')[0]
    version_eval_dataset = get_version(dataset_path)

    amp_new = 0
    if 'amp_new' in dataset_path:
        amp_new = 1

    if 'M=rand' in weights_path:
        if 'amp_new' in dataset_path:
            return _serialize(weights_snr=weights_snr,
                              weights_version=weights_version,
                              number_of_sources=number_of_sources,
                              diff_method=diff_method,
                              T=snapshots,
                              array_form=array_form,
                              examples_in_eval_data=examples_in_eval_data,
                              snr_eval_dataset=snr_eval_dataset,
                              version_eval_dataset=version_eval_dataset,
                              amp_new=amp_new)
        else:
            return _serialize(weights_snr=weights_snr,
                              weights_version=weights_version,
                              number_of_sources=number_of_sources,
                              diff_method=diff_method,
                              T=snapshots,
                              array_form=array_form,
                              examples_in_eval_data=examples_in_eval_data,
                              snr_eval_dataset=snr_eval_dataset,
                              version_eval_dataset=version_eval_dataset)

    elif 'M=rand' not in weights_path:
        if 'amp_new' in dataset_path:
            return _serialize(weights_snr=weights_snr,
                              weights_version=weights_version,
                              number_of_sources=number_of_sources,
                              diff_method=diff_method,
                              T=snapshots,
                              array_form=array_form,
                              examples_in_eval_data=examples_in_eval_data,
                              snr_eval_dataset=snr_eval_dataset,
                              version_eval_dataset=version_eval_dataset,
                              number_of_sources_for_train=number_of_sources_for_train,
                              amp_new=amp_new)
        else:
            return _serialize(weights_snr=weights_snr,
                              weights_version=weights_version,
                              number_of_sources=number_of_sources,
                              diff_method=diff_method,
                              T=snapshots,
                              array_form=array_form,
                              examples_in_eval_data=examples_in_eval_data,
                              snr_eval_dataset=snr_eval_dataset,
                              version_eval_dataset=version_eval_dataset,
                              number_of_sources_for_train=number_of_sources_for_train)


def serialize_non_trainable(dataset_path, method='CRLB'):
    """ serialization of non-trainable methods like CRLB or music with ula 7"""
    array_form, N, T, snr, M, size, version =\
        get_param_from_dataset(
            dataset_path, 'array_form', 'n', 't', 'snr', 'm', 'size', 'version')
    if 'CRLB' in method:
        return _serialize(method=f'{method}', array_form=array_form, N=N, T=T, SNR=snr, M=M, size=size, version=version)
    elif method == 'music_ULA_7':
        return "method_music_ULA-7_" + _serialize(array_form=array_form, N=N, T=T, SNR=snr, M=M, size=size, version=version)


def serialize_dataset(version, array_form, size, number_of_sources, snapshots, snr):
    """returns the serialization of a dataset with these parameters"""
    return (f"{version}_{array_form}_Far_field_NarrowBand_non-coherent_{size}_M={number_of_sources}_N=7_T="
            f"{snapshots}_SNR={snr}_eta=0_sv_noise_var0_bias=0_.h5")


def check_dataset_exists(version, array_form, size, number_of_sources, snapshots, snr, train_test_ratio):
    """ check if dataset exist according to the input parameters"""
    train_datasets_dir = get_path_config()['train_datasets_path']
    test_dataset_dir = get_path_config()['test_datasets_path']
    train_serialization = serialize_dataset(
                                          version, array_form, size, number_of_sources, snapshots, snr)
    test_serialization = serialize_dataset(
                                          version, array_form, int(size * train_test_ratio), number_of_sources, snapshots, snr)
    train_dataset_path = os.path.join(train_datasets_dir, train_serialization)
    test_dataset_path = os.path.join(test_dataset_dir, test_serialization)
    if (os.path.isfile(train_dataset_path)) and (os.path.isfile(test_dataset_path)):
        return True, train_dataset_path, test_dataset_path
    else:
        return False, False, False


def get_version(path_to_name):
    """ returns the version according to some string (path or else)"""

    if path_to_name.find('new_version') > -1:
        return 'new_version'
    elif path_to_name.find('optimal') > -1:
        return 'optimal-version'
    else:
        return 'test_case'


def get_array_form(path_to_dataset:str):
    """ return the array form of the input dataset path"""
    b = path_to_dataset.find('MRA')
    if b > -1:
        return path_to_dataset[b: b+5]
    b = path_to_dataset.find('ULA')
    if b > -1:
        return path_to_dataset[b: b+5]


def get_snapshots(path_to_weights):
    """ returns number of snapshots in the train of this weights"""
    b = path_to_weights.find('T=')
    if b == -1:
        return '100'
    elif b > -1:
        return path_to_weights.split('T=')[1].split("_")[0]


def get_param_from_dataset(dataset_path: str, *params):
    """
    given a path to a dataset and a bunch of parameters, it returns a tuple
     of the parameters values if exist in the dataset path (the path is kind of serialization)
    """
    list_to_return = []
    dataset_path_lower = dataset_path.lower()

    for param in params:
        param_lower: str = param.lower()
        if dataset_path_lower.find(f"{param_lower}=") > -1:
            mid = dataset_path_lower.split(f"{param_lower}=")[1].split("_")[0]

        elif param_lower.startswith('field'):
           mid = 'Far_field' if 'far_field' in dataset_path_lower else 'Near_field'
        elif param_lower == 'signal_nature':
            mid = 'non-coherent' if 'non-coherent' in dataset_path_lower else 'coherent'
        elif param_lower == 'signal_type':
            mid = 'NarrowBand' if 'narrow' in dataset_path_lower else 'BroadBand'
        elif param_lower == 'sv_noise_var':
            mid = dataset_path_lower.split('sv_noise_var')[1][0]

        elif param == 'array_form':
            b = dataset_path.find('MRA')
            if b == -1:
                b = dataset_path.find('ULA')
            mid = dataset_path[b: b + 5]

        elif param == 'version':
            b = dataset_path.find('new_version')
            if b == -1:
                b = dataset_path.find('optimal')
                if b == -1:
                    mid = 'test'
                else:
                    mid = 'optimal'
            else:
                mid = 'new_version'

        elif param == 'size':
            mid = dataset_path.split('coherent_')[1]
            mid = mid.split('_')[0]

        else:
            my_param = param.upper()
            if dataset_path.find(f"{my_param}=") > -1:
                mid = dataset_path.split(f"{my_param}=")[1].split("_")[0]
            elif dataset_path.find(f"{param}=") > -1:
                mid = dataset_path.split(f"{param}=")[1].split("_")[0]
        try:
            if mid.isdigit():
                mid = int(mid)
            list_to_return.append(mid)
            mid = None
        except NameError:
            raise NameError(f"there is no such parameter {param} in the supplied dataset")
    if len(list_to_return) == 1:
        return list_to_return[0]
    else:
        return tuple(list_to_return)


def save_json(my_dict, my_dir, json_name):
    my_json_path = translate_path(os.path.join(my_dir, f"{json_name}.json"))
    with open(my_json_path, 'w') as f:
        json.dump(my_dict, f)


def load_json(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)
    return data


def power(sig):
    """ power of array"""
    return np.mean(np.abs(sig)**2)


def rms(array):
    """ given array of real numbers, returns the rms of it"""
    if type(array)!= np.array:
        array = np.array(array)
    return np.sqrt(np.mean(array ** 2))


def plot_angle_diff(s1=None, s2=None, *args, **kwargs):
    """create a plot of differences of angles arrays.
     * args is for giving more arrays s3, s4, etc.
      ** kwargs is for 'unwrap' yes or no"""
    list_of_diffs = [np.angle(s2) - np.angle(s1)]
    for sig in args:
        list_of_diffs.append(np.angle(sig) - np.angle(s1))
    for obj in list_of_diffs:
        unwrap = kwargs.get('unwrap')
        plt.plot(np.unwrap(obj)) if unwrap else plt.plot(obj)
    plt.show()


def get_list_of_doa(list_of_numbers):
    """ the content of this function is not useful.
     it's used only as a block when generating the 'y' in the test data
      where it has to be created as part of the code,
      however y in test is the ground truth from the experiment"""
    dict_of_locations = {1: -30, 2: -10, 3: 0, 4: 20, 5: 40, 6: 60}
    list_of_doa = [dict_of_locations[key] for key in list_of_numbers]
    return list_of_doa


def estimate_snr(signal, window_size):
    """this function estimates signal-to-noise ratio of a fixed signal held in numpy array"""

    smooth_signal = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    noise = signal - smooth_signal

    return 10 * np.log10(power(signal) / power(noise))


def minimize_arrays_diff(array_1, array_2):
    """given two arrays, this function returns the float that minimize their diff"""
    array_1 = np.array(array_1)
    array_2 = np.array(array_2)
    diff = array_1 - array_2
    s_optimal = np.median(diff)
    return s_optimal


def find_best_array_index(list_of_arrays, ref_array):
    """given list of arrays and a reference array, this function returns
    the index of the "closest" array in abs to the reference"""
    best_diff = np.inf
    best_index = 0
    for i, array in enumerate(list_of_arrays):
        s_optimal = minimize_arrays_diff(array, ref_array)
        diff = np.sum(np.abs(array - s_optimal - ref_array))
        if diff < best_diff:
            best_diff = diff
            best_index = i
    return best_index

