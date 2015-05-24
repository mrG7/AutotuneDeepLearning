__author__ = 'Roman'

import numpy as np

def pause():
    raw_input("PRESS ENTER TO CONTINUE.")

def gen_vects_to_mut(num_total_neurons):
    A_set_of_nnets = np.zeros((num_total_neurons,num_total_neurons), dtype=int)
    A_set_of_nnets[0] = np.array([0]+[1]*(num_total_neurons-1))

    for i in xrange(1, num_total_neurons):
        A_set_of_nnets[i] = np.roll(A_set_of_nnets[i-1],1)

    return A_set_of_nnets

def gen_subset_F(S):
    Smax = np.max(S)
    N = len(S)
    # print Smax

    temp = np.zeros(N)
    p_array = np.zeros(N)

    for i in xrange(N):
        temp[i] = np.exp(-float(S[i])/Smax)

    for i in xrange(N):
        p_array[i] = temp[i]/np.sum(temp)
    print "p_array: ", p_array

    # we allow only 25% of the network to mutate
    # p = np.around(N*0.25)
    p = 4

    while True:
        # rand_vectors = np.random.random()
        rand_vectors = np.random.random(p)
        # print "rand_vectors: ", rand_vectors

        bins = np.sort(p_array)
        # print "bins: ", bins

        # indexes determine which vectors we will pick for the mutation
        inds = np.digitize(rand_vectors, bins)
        # print inds # but this is the number of bins

        if len(np.unique(inds)) == p:
            break

    print "rand_vectors: ", rand_vectors
    print "bins: ", bins
    # print inds # but this is the number of bins
    # pause()

    inds -= 1 # minus 1 since the indexes of our vectors begin from 0

    return inds

def calc_lagrang(hessianMat, W_matrix_j, j_num):
    # hessianInverse = np.linalg.inv(hessianMat)
    hessianInverse = np.matrix(hessianMat)
    Lagrang = np.dot(W_matrix_j,W_matrix_j).astype(float) / float(2*hessianInverse[j_num, j_num])
    return Lagrang

def cross_mutate(F_subset_of_nnets):
    print "\nVectors to cross & mutate:"
    print F_subset_of_nnets
    n_vectors = len(F_subset_of_nnets)
    elems_in_vect = len(F_subset_of_nnets[0])
    m_pairs = len(F_subset_of_nnets)/2 # number of pairs needed to form

    # randomly generate indexes of pairs (just list of unique ids)
    rand_id_arr = np.arange(n_vectors)
    np.random.shuffle(rand_id_arr)
    print "Random indexes of elements to form pairs:"
    print rand_id_arr[:n_vectors]

    idx_for_pairs = rand_id_arr[:n_vectors]

    all_pairs_list = []
    counter = 0
    # unite vectors in pairs
    for i in xrange(m_pairs):
        test = np.array([ F_subset_of_nnets[idx_for_pairs[counter]],
                          F_subset_of_nnets[idx_for_pairs[counter+1]] ])
        counter += 2
        all_pairs_list.append(test)

    print "\nInitial pairs: "
    print all_pairs_list

    F_new_subset = []

    # crossing for each pair
    for i in xrange(m_pairs):
        print "\nPair %s:" % (i+1)
        print all_pairs_list[i]
        cross_id = np.random.choice(elems_in_vect-1)
        print "Crossing id (by counting): ", cross_id+1
        pair_elem_1 = np.append(all_pairs_list[i][0][:cross_id],
                        all_pairs_list[i][1][cross_id:])
        pair_elem_2 = np.append(all_pairs_list[i][1][:cross_id],
                        all_pairs_list[i][0][cross_id:])
        print np.array([pair_elem_1,pair_elem_2])

        # modification
        mut_id = np.random.choice(elems_in_vect-1)
        print "Modification id (by counting): ", mut_id+1
        pair_elem_1[mut_id] = not pair_elem_1[mut_id]
        pair_elem_2[mut_id] = not pair_elem_2[mut_id]
        new_pair = np.vstack([pair_elem_1, pair_elem_2])
        print new_pair

        F_new_subset.append(new_pair)

    print "\nResult crossed and mutated pairs:"
    print F_new_subset

    return F_new_subset

def modify_W_matrix(W_matrix, vec_of_neurons):
    print "\nPattern of modification: ", vec_of_neurons
    modified_W = W_matrix
    for i in np.where(vec_of_neurons == 0)[0]:
        # print "Iteration %s:" %i
        W_matrix_transposed = np.transpose(W_matrix)

        if vec_of_neurons[i] == 0:
            W_matrix_transposed[i] *= 0

        W_matrix = np.transpose(W_matrix_transposed)
        modified_W = W_matrix

    return modified_W

#TODO: calc hessian Matrix - theano.gradient.hessian()

# Generate population of vectors for further processing
A_set_of_nnets = gen_vects_to_mut(6)
print "Full generation:"
print A_set_of_nnets

#TODO: calc errors for every generation (lagranges)

#TODO: pass the array of errors (cost, negative_log_likelihood)

inds = gen_subset_F([1,2,3,5,7,9]) # pass the array of errors for every (sum of lagrangians)
F_new_subset = cross_mutate(A_set_of_nnets[inds])

# test_W_matrix = np.array([np.arange(6),np.arange(6)])
#TODO: pass real W_matrix

test_W_matrix = np.ones(shape =(6,6))
print "\nInitial weight matrix:"
print test_W_matrix

error_results = []
# print F_new_subset[0][0]
for i in xrange(len(F_new_subset)):
    for j in xrange(2):

        new_W_matrix = modify_W_matrix(np.copy(test_W_matrix), F_new_subset[i][j])
        print "Modified matrix W:"
        print new_W_matrix
        pause()
        #TODO: initiate network with modified_W
        #TODO: calc the error function for every generation
        # error_cur_vector = 0.5 # CALCULATION!!!
        # error_results.append(error_cur_vector)

# val, idx = min((val, idx) for (idx, val) in enumerate(my_list))

#TODO: find the minimum value of the error function and the corresponding neurons vector

#TODO: plot the error function vs number of neurons
#TODO: plot the final parameter structure

# a = [1, 2, 3]
# hessianMat = [[1, 2], [3, 4]]
# calc_lagrang(hessianMat, a, 1)
# print np.dot(a,b)

