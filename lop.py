import numpy as np
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

def find_spectrum(W):
    # for a given observation matrix W returns the spectrum
    # and the numpy array of solutions which maximize the consistency
    st = time.time()
    M, N = W.shape
    T = W.sum()  ## total number of comparisons
    Iset = list(range(0, M))      #set of all indices 0,...M-1
    ## enumerate all combinations of k-subsets (list of lists of tuples)
    II = [list(combinations(Iset, k)) for k in range(M+1)]

    # initialize the spectrum for k=1
    k = 1
    L = len(II[k])
    spec_tmp = np.zeros((L, T+1))
    spec_tmp[:, 0] = 1
    order_tmp = np.zeros((L, 1))
    Spectrum = [None, dict(zip(II[k], spec_tmp))]   ## element number k in the list, fill-up with None on position 0
    Order = [{(): 0}, dict(zip(II[k], order_tmp))]  # order of en empty set = 0 (for convenience)
    # recursive calculation of the spectrum for k=2,...,M
    for k in range(2, M+1):
        L = len(II[k])
        spec_tmp = np.zeros((L, T + 1)) # initialize the spectrum for k
        order_tmp = np.zeros((L, 1)).astype(int)    # order of the spectrum polynomial
        for l, ii_l in enumerate(II[k]):
            w = W[ii_l, :][:, ii_l]        # submatrix indexed with all elements in II^[k]_l
            u_power = w.sum(axis=1)
            order_past = np.zeros(k)
            for i in range(k):
                ii_index = list(ii_l)
                ii_index.pop(i)                     # remove the index i from the list
                tuple_index = tuple(ii_index)       # transform to tuple (necessary for indexing the dictionary)
                Guk1j = Spectrum[k-1][tuple_index]              # subspectrum from the set k-1
                delta = u_power[i]
                spec_tmp[l, delta:] += Guk1j[:T + 1 - delta]
                order_past[i] = Order[k-1][tuple_index]
            order_tmp[l] = (order_past+u_power).max()
        Spectrum.append(dict(zip(II[k], spec_tmp)))
        Order.append(dict(zip(II[k], order_tmp)))
    et = time.time()
    do_print_time = True
    if do_print_time:
        print('forward phase execution time:', et-st, 'seconds')

    st = time.time()
    #  finding solution(s) of the problem via backward recursion
    get_solutions = True
    if get_solutions:
        A = [{} for n in range(M+1)]        # list of M lists
        A[M] = dict(zip(II[M], [-np.ones((1,M)).astype(int)]))  # ranking prototype filled with -1
        for k in range(M, 0, -1):
            La = len(A[k])
            for l, ii_l in enumerate(A[k]):
                w = W[ii_l, :][:, ii_l]  # submatrix indexed with all elements in II^[k]_l
                u_power = w.sum(axis=1)
                for i in range(k):
                    ii_index = list(ii_l)
                    ii_index.pop(i)  # remove the index i from the list
                    tuple_index = tuple(ii_index)  # transform to tuple (necessary for indexing the dictionary)
                    if Order[k][ii_l] == Order[k - 1][tuple_index] + u_power[i]: # order-compatible solutions
                        solutions_tmp = A[k][ii_l].copy()
                        for j in range(solutions_tmp.shape[0]):
                            solutions_tmp[j][M-k] = ii_l[i]      # add the order-compatible index

                        if tuple_index in A[k-1]:       ## append
                            A[k-1][tuple_index] = np.concatenate((A[k-1][tuple_index], solutions_tmp), axis=0)
                        else:                           ## create
                            A[k-1][tuple_index] = solutions_tmp
    et = time.time()
    do_print_time = True
    if do_print_time:
        print('backward phase execution time:', et - st, 'seconds')

    order = A[0][()]  # this is the numpy array
    return spec_tmp, order

def find_spectrum_permutations(W):
    # for a given observation matrix W returns the spectrum
    # and the list of solutions which maximize the consistency
    st = time.time()
    M, N = W.shape
    T = W.sum()
    Iset = list(range(0, M))
    II = list(permutations(Iset))
    triu_ind = np.triu_indices(M, k=1)
    spec_tmp = np.zeros(T+1)
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
    M= 9
    p_real = 0.6
    K = 1   # number of comparisons per pair; increasing K will increase the number of equivalent solutions
    print("generate random {}x{} observation matrix for {} total pairwise comparisons:".format(M, M, K*M*(M-1)//2))
    W = random_W(M, p_real, K=K)
    print("this is the matrix:")
    print(W)
    T = W.sum()
    print("calculate the spectrum and find all the solutions with our algorithm")
    spectrum, order = find_spectrum(W)
    print("calculate the spectrum and find all the solutions by brute force (may be very long for M>10!!)")
    spectrum_perm, order_perm = find_spectrum_permutations(W)
    print("compare the results:")
    print("algorithmic spectrum:")
    print(spectrum)
    print("brute-force spectrum:")
    print(spectrum_perm)
    print("algorithmic solutions:")
    print(order)
    print("brute force solutions (may be listed in different order then those obtained algorithmically):")
    print(order_perm)



if __name__ == "__main__":
    main()