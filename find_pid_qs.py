# Find alternative joint distributions to satisfy Delta_P conditions
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import fxn_library as fxn


# Define P over 3 variables
events = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# pmf = np.transpose(np.array([8*[0.125]]))  # Uniform distribution example
pmf = np.transpose(np.array([[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3]]))  # A canonical random example

# Generate 100 random pmfs
for j in range(100):
    pmf = fxn.generate_random_pmf(8)  # Some random example
    pmf = np.transpose(np.array([[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3]]))  # A canonical random example #FIXME: RMEOVE

    # Convert to dataframe
    pmf_stk = np.hstack((events, pmf))
    pmf_df = pd.DataFrame(data=pmf_stk, columns=["X", "Y", "Z", "p"])

    # FIXME: REMOVE LATER
    print('\n\n')
    print(pmf_df)

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
        cmi_arr, kl_arr, uim = fxn.compute_unique_info(pmf_df, null, [t, s1, s2], res_lower, res_upper, delta_q=0.01,
                                                       verbose=False, get_metadata=True)

        # FIXME: COMPUTE I(X;Z), I(Y;Z)

        # Output unique information
        print("*** UI = ", uim.min_cmi)
        print(uim.min_q)

        # Plotting
        plt.scatter(kl_arr, cmi_arr, marker=".")
        plt.xlabel("KL divergence")
        plt.ylabel("I(" + t + "; " + s1 + " | " + s2 + ")")
        plt.title("UI(" + t + "; " + s1 + " \\ " + s2 + ")")
        plt.show()

    print("Stop")

print("Placeholder")
