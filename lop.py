import numpy as np
import math
from itertools import combinations, permutations
import time

def random_W(M, p, K=1):
    ### generate random entries for Mx M matrix
    # M size
    # p probablity of A winning over B  when A is ranked before B
    # K = 1  # number of games per pair
    W = np.zeros((M,M)).astype(int)
    ii = np.triu_indices(M, k=1)
    Ntriu = M*(M-1)//2
    W[ii] = np.random.choice(a=np.array([0, 1]), size=(Ntriu, K), p=[1-p, p]).sum(axis=1)
    W -= np.triu(W).T
    W[np.tril_indices(M, k=-1)] += K

    return W

def find_spectrum(W, example1=False):
    # for a given observation matrix W returns the spectrum
    # cannot find the solutions

    st = time.time()
    M, N = W.shape
    T = W.sum()  ## total number of comparisons
    Iset = list(range(0, M))      #set of all indices 0,...M-1
    # initialize the spectrum for k=1
    k = 1
    FF = list(combinations(Iset, k))    # this is family F_1 (line 3 of Algorithm 1)
    L = M       # L = len(FF) = |F_1|
    spec_tmp = np.zeros((L, T+1)).astype(int)
    spec_tmp[:, 0] = 1                  # (line 5 of Algorithm 1)
    Spectrum = dict(zip(FF, spec_tmp))   ## element number k in the list, fill-up with None on position 0
    # recursive calculation of the spectrum for k=2,...,M
    for k in range(2, M+1):             # (line 8 of Algorithm 1)
        if example1:
            print("Calculation for the subsets in the family F_{}".format(k))
        FF = list(combinations(Iset, k))    # this is family F_k (line 9 of Algorithm 1)
        L = math.comb(M, k)                 # L = len(FF) = |F_k|
        spec_tmp = np.zeros((L, T + 1)).astype(int) # initialize the spectrum for k (line 11 of Algorithm 1)
        for l, ii_l in enumerate(FF):       # ii_l is the same as I', (line 10 of Algorithm 1)
            w = W[ii_l, :][:, ii_l]         # submatrix indexed with all elements in FF^[k]_l
            u_power = w.sum(axis=1)         # these are d_e(I')
            for i in range(k):              # (line 12 of Algorithm 1)
                ii_index = list(ii_l)
                ii_index.pop(i)                     # remove the index i from the list
                tuple_index = tuple(ii_index)       # transform to a tuple (necessary for indexing the dictionary)
                if example1:
                    print("path connecting subset I={} to subset I\i={}, is labeled with C_i(I)={}, i={}".
                          format(np.array(ii_l)+1, np.array(ii_index)+1, u_power[i], ii_l[i]+1))
                Guk1j = Spectrum[tuple_index]              # subspectrum from the set k-1
                delta = u_power[i]
                spec_tmp[l, delta:] += Guk1j[:T + 1 - delta]    # (line 13 of Algorithm 1)
            if example1:
                print("=> spectrum at the node {} = {}".format(np.array(ii_l)+1, spec_tmp[l,:]))
        Spectrum = dict(zip(FF, spec_tmp))

    et = time.time()
    do_print_time = True
    if do_print_time:
        print('Execution time:', et-st, 'seconds')

    return spec_tmp                                 # (line 17 of Algorithm 1)

def find_solutions(W, example2=False):
    # for a given observation matrix W returns the numpy array of solutions which maximize the consistency
    st = time.time()
    M, N = W.shape
    T = W.sum()  ## total number of comparisons
    Iset = list(range(0, M))  # set of all indices 0,...M-1
    k = 1
    FF = list(combinations(Iset, k))  # this is family F_1
    L = M  # L = len(FF) = |F_1|
    order_tmp = np.zeros((L, 1)).astype(int)        #
    Degree = [{(): 0}]                       # order of en empty set = 0 (for convenience)
    Degree.append(dict(zip(FF, order_tmp)))  # (line #5 of Algorithm 2)
    # recursive calculation of the degree for k=2,...,M
    if example2:
        print("forward recursion: calculation of the degree of the polynomials")
    for k in range(2, M+1):                 # (line #8 of Algorithm 2)
        FF = list(combinations(Iset, k))      # create the family F_k  (line #9 of Algorithm 2)
        L = math.comb(M, k)     # number of subsets of I of length k  <= len(FF)
        order_tmp = np.zeros((L, 1)).astype(int)  # initialize order of polynomial
        for l, ii_l in enumerate(FF):
            w = W[ii_l, :][:, ii_l]        # submatrix indexed with all elements in FF^[k]_l
            d_e_I = w.sum(axis=1)         # these are d_e(I') for all e in I':
            order_past = np.zeros(k).astype(int)
            for i in range(k):
                ii_index = list(ii_l)
                ii_index.pop(i)                     # remove the index i from the list
                tuple_index = tuple(ii_index)  # transform to tuple (necessary for indexing the dictionary)
                order_past[i] = Degree[k-1][tuple_index]
            order_tmp[l] = (order_past+d_e_I).max()     # (line #11 of Algorithm 2)
            if example2:
                print("=> degree at the node {} = {}".format(np.array(ii_l)+1, order_tmp[l]))
        Degree.append(dict(zip(FF, order_tmp)))
    et = time.time()
    do_print_time = True
    if do_print_time:
        print('forward phase execution time:', et-st, 'seconds')

    st = time.time()
    #  finding solution(s) of the problem via backward recursion
    if example2:
        print("backward recursion: finding the solutions")
    R_hat = [{} for n in range(M+1)]            # list of M dictionaries  # (line #21  of Algorithm 2, for all k)
    # Note: the subfamilies A_k will be define implicitly by adding indices/tuples to dictionnary R_hat[k]
    R_hat[M] = dict(zip(FF, [-np.ones((1,M)).astype(int)]))  # (line #16 of Algorithm 2) ranking prototype; "-1" means that we don't know the value
    for k in range(M, 0, -1):               # (line #20 of Algorithm 2)
        for l, ii_l in enumerate(R_hat[k]):     # (line #22 of Algorithm 2)
            w = W[ii_l, :][:, ii_l]         # submatrix indexed with all elements in FF^[k]_l
            d_e_I = w.sum(axis=1)
            for i in range(k):
                ii_index = list(ii_l)
                ii_index.pop(i)  # remove the index i from the list
                tuple_index = tuple(ii_index)  # transform to tuple (necessary for indexing the dictionary)
                if Degree[k][ii_l] == Degree[k - 1][tuple_index] + d_e_I[i]: # (line #25 of Algorithm 2)
                    R_hat_tmp = R_hat[k][ii_l].copy()       # (line #26 of Algorithm 2)
                    for j in range(R_hat_tmp.shape[0]):
                        R_hat_tmp[j][M-k] = ii_l[i]         # (line #28 of Algorithm 2) add the degree-compatible index to the solutions
                    # below we implement (lines #30-35 of Algorithm 2)
                    if tuple_index in R_hat[k-1]:           # (line #33 of Algorithm 2)
                        R_hat[k-1][tuple_index] = np.concatenate((R_hat[k-1][tuple_index], R_hat_tmp), axis=0) # (line #31 of Algorithm 2)
                    else:                           ## create
                        R_hat[k-1][tuple_index] = R_hat_tmp     # (lines #33 and #34 of Algorithm 2),
                    if example2:
                        print("=> solution at the node labeled with {};{} = \n"
                              "{}".format(np.array(ii_index) + 1, Degree[k - 1][tuple_index], R_hat_tmp+1))
    et = time.time()
    do_print_time = True
    if do_print_time:
        print('backward phase execution time:', et - st, 'seconds')

    order = R_hat[0][()]  # (lines #39 of Algorithm 2)
    return order

def find_spectrum_permutations(W):
    # for a given observation matrix W returns the spectrum
    # and the list of solutions which maximize the consistency
    st = time.time()
    M, N = W.shape
    T = W.sum()
    Iset = list(range(0, M))
    II = list(permutations(Iset))
    triu_ind = np.triu_indices(M, k=1)
    spec_tmp = np.zeros(T+1).astype(int)
    order_list = []
    max_order = 0
    for ii_l in II:
        w = W[ii_l, :][:, ii_l]
        C = w[triu_ind].sum()
        spec_tmp[C] += 1
        if C == max_order:
            order_list.append(ii_l)
        elif C > max_order:
            order_list =[ii_l]
            max_order = C
    et = time.time()
    print('brute-force execution time:', et - st, 'seconds')
    return spec_tmp, np.array(order_list)

def main():
    # Example 1
    show_example_1 = True
    if show_example_1:
        print("-----------------------------------------------------------")
        print("EXAMPLE 1: steps of the algorithm used to find the spectrum")
        W = np.array([[0,1,1,0],[0,0,1,1],[0,0,0,0],[1,0,1,0]])
        print("this is the matrix:")
        print(W)
        print("the description of the graph from Fig.1 is shown below:")
        find_spectrum(W, example1=show_example_1)
    show_example_2 = True
    if show_example_2:
        print("-----------------------------------------------------------")
        print("EXAMPLE 2: steps of the algorithm used to find all solutions")
        W = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]])
        print("this is the matrix:")
        print(W)
        print("the description of the graph from Fig.2 is shown below ('0' in the solution means that the order is not yet undefined):")
        find_solutions(W, example2=show_example_2)

    M = 9
    p_real = 0.7
    K = 1   # number of comparisons per pair; increasing K will increase the number of equivalent solutions
    print("-----------------------------------------------------------")
    print("generate random {}x{} observation matrix for {} total pairwise comparisons:".format(M, M, K*M*(M-1)//2))
    W = random_W(M, p_real, K=K)
    print("this is the matrix:")
    print(W)
    T = W.sum()
    print("calculate the spectrum with our algorithm")
    spectrum = find_spectrum(W)
    print("calculate the spectrum and find all solutions by brute force (for M>10, it may be very long !!)")
    spectrum_perm, order_perm = find_spectrum_permutations(W)
    print("compare the results:")
    print("algorithmic spectrum:")
    print(spectrum)
    print("brute-force spectrum:")
    print(spectrum_perm)

    print("find all the solutions with our algorithm")
    order = find_solutions(W)
    print("compare the results:")
    print("algorithmic solutions:")
    print(order)
    print("brute force solutions (may be listed in different order then those obtained algorithmically):")
    print(order_perm)



if __name__ == "__main__":
    main()