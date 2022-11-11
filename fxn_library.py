import numpy as np
import pandas as pd


# Define constants
zero_threshold = 1e-20


# ########################################### FUNCTIONS ###########################################
# Input:
# joint - dataframe for joint distribution between XYZ
# var_lst - list of variable names that should form the marginal
def compute_marginal(joint, var_lst):
    df_marg = joint.groupby(var_lst)['p'].sum()
    prob = df_marg.values                                   # Marginal probabilities
    vals_series = df_marg.index.values                      # Values of random variables (as series)
    vals = np.array([np.array(v) for v in vals_series])     # Values of random variables (as nparray)

    if vals.ndim == 1:
        vals = np.expand_dims(vals, axis=1)

    marginal = pd.DataFrame(np.hstack((vals, np.expand_dims(prob, axis=1))))

    # Name the columns
    var_lst.append('p')
    marginal.columns = [var_lst]

    return marginal


# Input:
# n - dimension (number of events in distribution)
def generate_random_pmf(n):
    pmf = np.random.random((1, n))
    pmf = pmf / np.sum(pmf)
    return np.transpose(pmf)


# Inputs P and Q assumed to be np arrays
def kl_divergence(p, q):
    num_events = q.shape[0]
    kl_div = 0

    for i in range(num_events):
        # Infinity protection: when denominator is zero and numerator is not
        if p[i] < zero_threshold < q[i]:
            return -1

        # NaN protection
        if p[i] < zero_threshold or q[i] < zero_threshold:
            continue

        kl_div += q[i] * np.log2(q[i] / p[i])

    return kl_div


# Generic conditional mutual information for I(X;Y|Z)
# Inputs:
# p - total joint distribution
# m0 - one variable marginal distribution
# m1 - pairwise marginal 1
# m2 - pairwise marginal 2
# var_lst - list of variable names. 1st one is var to condition on (1-var marginal), remaining are pairs
def conditional_mutual_info(p, m0, m1, m2, var_lst):
    v0 = var_lst[0]
    v1 = var_lst[1]
    v2 = var_lst[2]

    cond_mi = 0
    for i in range(p.shape[0]):
        # Collect the necessary probabilities
        p_xyz = p['p'][i]
        p_z = m0.loc[m0[v0].squeeze() == p[v0][i]]['p'].values[0][0]
        yz_val = p[[v0, v1]].iloc[i]
        p_xz = m1.loc[(m1[v0].squeeze() == yz_val[0]) & (m1[v1].squeeze() == yz_val[1])]['p'].values[0][0]
        xy_val = p[[v2, v0]].iloc[i]
        p_yz = m2.loc[(m2[v2].squeeze() == xy_val[0]) & (m2[v0].squeeze() == xy_val[1])]['p'].values[0][0]

        # NaN protection
        if p_z < zero_threshold or p_xyz < zero_threshold or p_xz < zero_threshold or p_yz < zero_threshold:
            continue

        cond_mi += p_xyz * np.log2(p_z*p_xyz / (p_xz*p_yz))

    return cond_mi


# Input
# p - total joint distribution of XYZ
# var_lst - list of variable names that should form the marginal
def build_a_mat(p, var_lst):
    m1 = compute_marginal(p, var_lst)
    A = np.zeros((m1.shape[0], p.shape[0]))

    for i in range(m1.shape[0]):
        c1 = p[var_lst[0]] == m1[var_lst[0]].squeeze()[i]  # Condition on var 1
        c2 = p[var_lst[1]] == m1[var_lst[1]].squeeze()[i]  # Condition on var 2
        ind = p.loc[c1 & c2].index.values
        A[i, ind] = 1

    return A


# FIXME: GENERALIZE FOR DIFFERENT SIZED NULL SPACES
def compute_unique_info(pmf_df, null, var_lst, res_lower, res_upper, delta_q=0.01, verbose=False, get_metadata=False):
    # Define variables for loop meta data
    cmi_arr = []
    kl_arr = []
    uim = UI_Metadata(pmf_df)

    # Collect the relevant variables
    t = var_lst[0]
    s1 = var_lst[1]
    s2 = var_lst[2]
    pmf = np.expand_dims(pmf_df['p'].to_numpy(), axis=1)
    events = pmf_df[['X', 'Y', 'Z']].to_numpy()

    # One-variable marginal (absolutely constrained)
    m_1 = compute_marginal(pmf_df, [s2])

    for a in np.arange(res_lower.x[0], res_upper.x[0], delta_q):
        for b in np.arange(res_lower.x[1], res_upper.x[1], delta_q):
            # Define Q distribution
            q = a * null[:, 0] + b * null[:, 1] + np.transpose(pmf)
            q = np.hstack((events, np.transpose(q)))
            q = pd.DataFrame(data=q, columns=["X", "Y", "Z", "p"])

            # Get pairwise marginals
            m_2a = compute_marginal(q, [t, s2])
            m_2b = compute_marginal(q, [s1, s2])

            # Conditional mutual information under Q
            cmi = conditional_mutual_info(q, m_1, m_2a, m_2b, [s2, t, s1])
            cmi_arr.append(cmi)

            # KL divergence between Q and original P
            kl_div = kl_divergence(pmf_df['p'].to_numpy(), q['p'].to_numpy())
            kl_arr.append(kl_div)

            # Determine distributions of smallest and largest CMI
            if get_metadata:
                if uim.max_cmi < cmi:
                    uim.update_max(q, cmi, kl_div)
                if uim.min_cmi > cmi:
                    uim.update_min(q, cmi, kl_div)

        if verbose:
            print(a)

    return cmi_arr, kl_arr, uim


# ########################################### CLASSES ###########################################
# Class for storing metadata about unique information for a given distribution
class UI_Metadata:

    def __init__(self, p):
        self.p = p

        self.min_cmi = 1 / zero_threshold
        self.max_cmi = -zero_threshold

        self.min_q = 0
        self.max_q = 0

        self.min_kl = 0
        self.max_kl = 0

    def update_max(self, q, cmi, kl):
        self.max_q = q
        self.max_cmi = cmi
        self.max_kl = kl

    def update_min(self, q, cmi, kl):
        self.min_q = q
        self.min_cmi = cmi
        self.min_kl = kl
