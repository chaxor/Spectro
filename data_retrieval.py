import numpy as np


def retrieve_XY(file_path):
    # XY data is read in from a file in text format
    file_data = open(file_path).readlines()

    # The list of strings (lines in the file) is made into a list of lists
    # while splitting by whitespace and removing commas
    file_data = [line.rstrip('\n').replace(
            ',',
            ' ').split() for line in file_data]

    # Remove empty lists, make into numpy array
    xy_array = np.array([_f for _f in file_data if _f])

    # Each line is searched to make sure that all items in the line are a
    # number
    where_num = np.array(list(map(is_number, xy_array)))

    # The data is searched for the longest contiguous chain of numbers
    contig = contiguous_regions(where_num)
    try:
        # Data lengths
        data_lengths = contig[:, 1] - contig[:, 0]
        # All maximums in contiguous data
        maxs = np.amax(data_lengths)
        longest_contig_idx = np.where(data_lengths == maxs)
    except ValueError:
        print('Problem finding contiguous data')
        return np.array([])
    # Starting and stopping indices of the contiguous data are stored
    ss = contig[longest_contig_idx]

    # The file data with this longest contiguous chain of numbers
    # Float must be cast to each value in the lists of the contiguous data and
    # cast to a numpy array
    longest_data_chains = np.array(
        [[list(map(float, n)) for n in xy_array[i[0]:i[1]]] for i in ss])

    # If there are multiple sets of data of the same length, they are added in
    # columns
    column_stacked_data_chain = np.hstack(longest_data_chains)
    return column_stacked_data_chain

# http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def is_number(s):
    try:
        np.float64(s)
        return True
    except ValueError:
        return False
