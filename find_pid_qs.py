# Find alternative joint distributions to satisfy Delta_P conditions
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import fxn_library as fxn
import csv
import os


# Define P over 3 variables
events = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# pmf = np.transpose(np.array([8*[0.125]]))  # Uniform distribution example
pmf = np.transpose(np.array([[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3]]))  # A canonical random example

# Loop control
resolution = 0.05
allow_plotting = False

min_cmi_arr = []
min_kl_arr = []
min_mi_arr = []

for p_idx in range(2):
    pmf = fxn.generate_random_pmf(8)  # Some random example

    # FIXME: REMOVE
    # pmf = np.transpose(np.array([[0.029830, 0.062646, 0.165718, 0.016642, 0.121789, 0.287358, 0.133246, 0.182771]]))

    # Convert to dataframe
    pmf_stk = np.hstack((events, pmf))
    pmf_df = pd.DataFrame(data=pmf_stk, columns=["X", "Y", "Z", "p"])

    print(pmf_df)  # FIXME: REMOVE IF NOT ITERATING OVER RANDOM PMFS

    # Compare UI(Y; X\Z) to UI(X; Y\Z)
    ui_order = [["X", "Y", "Z"], ["Y", "X", "Z"]]

    # Save and write distributions and features to file
    pmf_df_targ = []
    feat_df = pd.DataFrame(columns=["KL", "MI", "CMI"])

    # Get metrics from original distribution P
    s2 = "Z"
    t = "Y"
    s1 = "X"
    m0 = fxn.compute_marginal(pmf_df, [s2])
    m1 = fxn.compute_marginal(pmf_df, [t, s2])
    m2 = fxn.compute_marginal(pmf_df, [s2, s1])
    # I(target; src1 | src2)
    cond_mi_orig = fxn.conditional_mutual_info(pmf_df, m0, m1, m2, [s2, t, s1])

    for i in range(len(ui_order)):
        print("Unique information for ", ui_order[i])

        # Select UI target and sources: UI(t; s1 \ s2)
        t = ui_order[i][0]
        s1 = ui_order[i][1]
        s2 = ui_order[i][2]

        # Build A matrix of marginal equations
        A1 = fxn.build_a_mat(pmf_df, [t, s1])
        A2 = fxn.build_a_mat(pmf_df, [t, s2])
        A = np.vstack((A1, A2))

        # Find general solution space
        A_exp = np.vstack((A, np.ones(A.shape[1])))
        null = null_space(A_exp)

        # Find positive solutions
        # Maximize(a+b) subject to sum of null space being positive
        c_upper = [-1, -1]
        c_lower = [1, 1]
        mat = -null

        res_upper = linprog(c_upper, A_ub=mat, b_ub=pmf, bounds=[(0, None), (0, None)])  # Max coef for positive prob
        res_lower = linprog(c_lower, A_ub=mat, b_ub=pmf, bounds=[(None, 0), (None, 0)])  # Min coef for positive prob

        nonzero_ui = ((res_upper.x[0] > res_lower.x[0]) and (res_upper.x[1] > res_lower.x[1]))  # Are there other slns?

        # Compute unique information
        if nonzero_ui:
            cmi_arr, kl_arr, mi_arr, uim = fxn.compute_unique_info(pmf_df, null, [t, s1, s2], res_lower, res_upper,
                                                        delta_q=resolution, verbose=True, get_metadata=True)

            # Output unique information
            print("*** UI = ", uim.min_cmi, uim.min_kl, uim.min_mi)
            print(uim.min_q)

            # Save minimizing distribution and features
            pmf_df_targ.append(uim.min_q['p'])
            feat_df.loc[len(feat_df)] = [uim.min_kl, uim.min_mi, uim.min_cmi]

            min_cmi_arr.append(uim.min_cmi)
            min_kl_arr.append(uim.min_kl)
            min_mi_arr.append(uim.min_mi)

            # Plot all found distributions: KL vs CMI
            if allow_plotting:
                plt.scatter(mi_arr, cmi_arr, marker=".")
                plt.xlabel("KL divergence")
                plt.ylabel("I(" + t + "; " + s1 + " | " + s2 + ")")
                plt.title("UI(" + t + "; " + s1 + " \\ " + s2 + ")")
                plt.show()

        # If there is no space of solutions, UI is zero FIXME: is it 0 or is it CMI under P?
        else:
            print("*** UI = ", 0)

            # Save minimizing distribution and features
            pmf_df_targ.append(pmf_df['p'])
            feat_df.loc[len(feat_df)] = [0, 0, 0]  # FIXME: CHECK THIS!!!

    # Save dataframes of distributions to csv
    for dfs in range(len(pmf_df_targ)):
        col_name = "targ_" + str(ui_order[dfs][0])
        pmf_df[col_name] = pmf_df_targ[dfs]
os.makedirs('C:\\Users\\amand\\Documents\\Research\\Project_Causal_PID\\yo', exist_ok=True)
pmf_df.to_csv('C:\\Users\\amand\\Documents\\Research\\Project_Causal_PID\\yo\\out.csv')

# File for CMI
with open('cmi.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(min_cmi_arr)
myfile.close()
# File for MI
with open('mi.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(min_mi_arr)
myfile.close()
# File for KL
with open('kl.csv', 'w') as myfile:
    wr = csv.writer(myfile)#, quoting=csv.QUOTE_ALL)
    wr.writerow(min_kl_arr)
myfile.close()

print("Placeholder")
