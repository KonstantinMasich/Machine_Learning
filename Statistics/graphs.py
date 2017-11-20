import matplotlib.pyplot as plt


def show_frequencies_hist(a, n_bins=1):
    plt.title('Frequencies histogram')
    plt.xlabel('values in bins')
    plt.ylabel('frequencies')
    plt.hist(a, bins=n_bins)
    plt.show()


def show_dot_plot(a):
    """
    Plots a dot plot graph for list a
    :param a: 1D list
    :return:
    """
    # 1. Counting all elements of a:
    a_elems = list(set(a))
    count_list = list(map(lambda x: a.count(x), a_elems))
    # 2. Plotting:
    plt.title('Dot plot')
    plt.xlabel('values')
    plt.ylabel('counts')
    plt.scatter(a_elems, count_list, color='black')
    plt.show()
