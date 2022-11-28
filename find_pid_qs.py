# Find alternative joint distributions to satisfy Delta_P conditions
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import fxn_library as fxn
import csv


# Define P over 3 variables
events = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# pmf = np.transpose(np.array([8*[0.125]]))  # Uniform distribution example
pmf = np.transpose(np.array([[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3]]))  # A canonical random example

min_cmi_arr = []
min_kl_arr = []
min_mi_arr = []

for p_idx in range(100):
    pmf = fxn.generate_random_pmf(8)  # Some random example

    # Convert to dataframe
    pmf_stk = np.hstack((events, pmf))
    pmf_df = pd.DataFrame(data=pmf_stk, columns=["X", "Y", "Z", "p"])

    print(pmf_df)  # FIXME: REMOVE IF NOT ITERATING OVER RANDOM PMFS

    # Compare UI(X; Y\Z) to UI(Y; X\Z)
    ui_order = [["X", "Y", "Z"], ["Y", "X", "Z"]]

    for i in range(len(ui_order)):
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

        # Compute unique information
        cmi_arr, kl_arr, mi_arr, uim = fxn.compute_unique_info(pmf_df, null, [t, s1, s2], res_lower, res_upper, delta_q=0.01,
                                                       verbose=False, get_metadata=True)

        # Output unique information
        print("*** UI = ", uim.min_cmi, uim.min_kl, uim.min_mi)
        print(uim.min_q)

        min_cmi_arr.append(uim.min_cmi)
        min_kl_arr.append(uim.min_kl)
        min_mi_arr.append(uim.min_mi)

    # Plotting
    # # 3D plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(kl_arr, mi_arr[:, 2], cmi_arr)
    # ax.set_xlabel("KL div")
    # ax.set_ylabel("Mutual info")
    # ax.set_zlabel("CMI")
    # plt.show()
    # # Normal plotting
    # plt.subplot(2, 1, i+1)
    # plt.scatter(mi_arr[:, 2], cmi_arr, marker=".")
    # plt.xlabel("KL divergence")
    # plt.ylabel("I(" + t + "; " + s1 + " | " + s2 + ")")
    # plt.title("UI(" + t + "; " + s1 + " \\ " + s2 + ")")
# plt.show()
        # # Plotting
        # plt.scatter(kl_arr, cmi_arr, marker=".")
        # plt.xlabel("KL divergence")
        # plt.ylabel("I(" + t + "; " + s1 + " | " + s2 + ")")
        # plt.title("UI(" + t + "; " + s1 + " \\ " + s2 + ")")
        # plt.show()

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
