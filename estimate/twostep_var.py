# Get the two-step m-estimation standard errors (Cameron & Trivedi, p.200)

import pandas as pd
import numpy as np
from scipy.special import factorial

import estimate





# Step 1 submatrices


def poissonl(step1_est, y, X):
    """ Compute the Poisson log likelihood for the given parameters.

    Args:
        step1_est (np.ndarray): the one-dimensional array of parameters.
        y (np.ndarray): The response variable.
        X (np.ndarray): The covariates.

    Returns:
        np.ndarray: The array of log-likelihood increments. Each row is an
            increment.
    """
    index = np.dot(X, step1_est)
    increments = - np.exp(index) + y * index - np.log(factorial(y))
    return increments


def poisson_score(step1_est, y, X):
    """ Get the scores of the Poisson log-likelihood.

    Returns:
        np.ndarray: The array of scores. Each row `i` is the score of increment
            `i` of the log-likelihood.
    """
    error = y - np.exp(np.dot(X, step1_est))
    return np.dot( np.diag(error), X)


def get_h1(step1_est, y, X):
    """ Get h1.
    """
    return poisson_score(step1_est, y, X)


def poisson_hess_sum(step1_est, y, X):
    """ Get the sum of Hessians of the Poisson log-likelihood

    Returns:
        np.ndarray: The sum of the hessians of each log-likelihood increment.
    """
    middle =  np.diag( - np.exp( np.dot(X, step1_est) ) )
    return np.linalg.multi_dot((np.transpose(X), middle, X))


def G_11(step1_est, y, X):
    """ Get G_11.
    """
    return (1.0 / len(y)) * poisson_hess_sum(step1_est, y, X)


def S_11(h1):
    """ Get S_11.
    """
    return (1.0 / len(h1)) * np.dot(np.transpose(h1), h1)




# Step 2 submatrices


def get_h2(step2_jac, step2_errors):
    """ Get h2.

    Args:
        step2_errors (np.ndarray): The (one-dimensional) array of differences
            between model and data trade shares.
    """
    return np.dot( np.diag(step2_errors), step2_jac )


def G_22(step2_jac):
    """ Gets the G_22 matrix.

    Args:
        step2_jac (np.ndarray): The jacobian of the step 2 estimation. It has
            one row for each i, j \neq i; and one column for each parameter
            (zeta not included).
    """
    #return np.dot(np.transpose(step2_jac), step2_jac)
    return (1.0 / len(step2_jac)) * np.dot(np.transpose(step2_jac), step2_jac)


def S_22(h2):
    """ Gets the S_22 matrix.
    Args:
        h2 (np.ndarray): the h2's...
    """
    #return np.dot( np.transpose(h2), h2 )
    return (1.0 / len(h2)) * np.dot( np.transpose(h2), h2 )




# Mixed submatrices


## S_12 and S_21

def mask(cities_to_keep, data):
    """ Get the boolean array to drop observations from iticount data.

    Args:
        cities_to_keep (list): The cities that we keep in the first stage.
            These include all known cities that made it after the lost cities
            were dropped.
    Returns:
        np.ndarray: A boolean array.
    """
    cond_i = data['id_i'] == cities_to_keep[0]
    cond_j = data['id_j'] == cities_to_keep[0]
    for c in range(1, len(cities_to_keep)):
        cond_i = cond_i | (data['id_i'] == cities_to_keep[c])
        cond_j = cond_j | (data['id_j'] == cities_to_keep[c])
    return cond_i & cond_j


def S_12(h1, masked_h2):
    """ Get S_12.

    Args:
        h1 (np.ndarray)
        masked_h2 (np.ndarray): The subset of the masked_h2 array associated
            to cities that made it to step 1.

    Returns:
        np.ndarray: The S_12 matrix.
    """
    assert len(h1) == len(masked_h2)
    return (1.0 / len(h1)) * np.dot( np.transpose(h1), masked_h2 )


def S_21(h1, masked_h2):
    """ Get S_21
    """
    return np.transpose( S_12(h1, masked_h2) )




## G_21, G_12


def G_12(step1_est, step2_est):
    shape = (len(step1_est), len(step2_est) - 2)
    return np.zeros(shape)


def numerical_grad(function, variables, parameters, h, e):
    """ Compute the numerical gradient of ``function`` at ``variables`` for step
        size `h`.

    Note it gives the derivative w.r.t. the first argument in ``variables``. This
    is the only one we need here.

    Args:
        function (function): The function to get the numerical gradient of. It
            can return a single number, a one-dimensional array, ...
        variables (np.ndarray): The one-dimensional argument on which to
            evaluate the function.
        parameters (np.ndarray): The one-dimensional argument that stays fixed
            for the function.
        h (float): The step size.

    Returns:
        np.ndarray: The gradient of ``function`` evaluated at ``variables`` for
            step size ``h``.
    """
    # Form inputs
    H = h * np.eye(len(variables))
    v_plus = variables + H
    v_minus = variables - H

    derivs = []
    for i in range(len(variables)):
        obj_plus = function(v_plus[i], parameters, e)
        obj_minus = function(v_minus[i], parameters, e)
        derivs.append((obj_plus - obj_minus) / (2 * h))

        break # Remove this to get the full derivative.

    return np.array(derivs)


def objective(step1_est, step2_est, e):
    """ The function to compute the numerical gradient on.
    """
    estimates = step2_est.copy()
    estimates[0] = step1_est[0]

    errors = e.get_errors(estimates)
    jac = e.get_jacobian(estimates, zeta_fixed = True)

    return get_h2(jac, errors)


def G_21(step1_est, step2_est, h, e):
    grads = numerical_grad(objective, step1_est, step2_est, h, e)
    grads = np.squeeze(grads)

    grads_sum = np.sum(grads, axis = 0)
    shape_zeros = (len(step2_est) - 2, len(step1_est) - 1)

    return (1.0 / len(grads)) * np.column_stack((grads_sum, np.zeros(shape_zeros)))




# Come together

def form_cov(G_inv, S):
    """ Form the variance covariance matrix.
    """
    return np.linalg.multi_dot((G_inv, S, np.transpose(G_inv)))


def twostep_cov(step1_est,
                step2_est,
                step1_data,
                cities_to_keep,
                h,
                cities_to_drop = [],
                cities_to_known = []):
    """ Compute the two-step variance covariance matrix. This is a wrapper.
    """
    y = step1_data['s_ij'].values
    X = step1_data.drop(['id_i', 'id_j', 's_ij'], axis = 1).values

    h1 = get_h1(step1_est, y, X)

    print(cities_to_drop)
    print(cities_to_known)
    e = estimate.EstimateAncient('directional',
                                 cities_to_drop = cities_to_drop,
                                 cities_to_known = cities_to_known)
    step2_jac = e.get_jacobian(step2_est, zeta_fixed = True)
    step2_errors = e.get_errors(step2_est)

    h2 = get_h2(step2_jac, step2_errors)

    # Form matrices
    g_11 = G_11(step1_est, y, X)
    g_12 = G_12(step1_est, step2_est)
    g_21 = G_21(step1_est, step2_est, h, e)
    g_22 = G_22(step2_jac)

    m = mask(cities_to_keep, e.df_iticount)
    masked_h2 = h2[m]

    s_11 = S_11(h1)
    s_12 = S_12(h1, masked_h2)
    s_21 = S_21(h1, masked_h2)
    s_22 = S_22(h2)

    print('g series:')
    print(g_11.shape)
    print(g_12.shape)
    print(g_21.shape)
    print(g_22.shape)

    print('s series:')
    print(s_11.shape)
    print(s_12.shape)
    print(s_21.shape)
    print(s_22.shape)

    G = np.block([[g_11, g_12],
                  [g_21, g_22]])
    print('--------------------')

    # Get dat G inverse
    g_11_inv = np.linalg.inv(g_11)
    g_22_inv = np.linalg.inv(g_22)
    G_inv = np.block([
        [g_11_inv, g_12],
        [- np.linalg.multi_dot((g_22_inv, g_21, g_11_inv)), g_22_inv]
        ])

    # Now S
    S = np.block([[s_11, s_12],
                  [s_21, s_22]])

    # Now form covariance matrix
    cov = (1.0 / len(e.df_iticount)) * form_cov(G_inv, S)

    # Keep only covariance matrix for zeta and step 2 estimates
    indices = [0] + [i for i in range(len(step1_est), len(cov))]

    print(cov.shape)
    cov = cov[indices, :]
    print(cov.shape)
    cov = cov[:, indices]
    print(cov.shape)

    return cov





# Now form interface to compute standard errors fast.

def get_cities_to_keep(step1_data):
    return step1_data['id_i'].unique().tolist()

def export(drop = None, known = None):
    to_twostep = './results/ancient/twostep/'

    for d in ['noneDrop/', 'qa01Drop/']:
        print(d)
        to_drop = []
        if d[:4] == 'qa01':
            to_drop = [d[:4]]

        step1_est = np.loadtxt(to_twostep + d + 'base/step1/coefs.csv')
        step2_est = np.loadtxt(to_twostep + d + 'base/optimum.csv')
        step1_data = pd.read_csv(to_twostep + d + 'base/step1/step1_data.csv')
        cities_to_keep = get_cities_to_keep(step1_data)

        cov = twostep_cov(step1_est,
                          step2_est,
                          step1_data,
                          cities_to_keep,
                          h = 1.0e-05,
                          cities_to_drop = to_drop)

        np.savetxt(to_twostep + d + 'base/twostep_var.csv', cov)

        for k in ['ma02', 'ha01', 'ma02ha01']:
            print(k)
            to_known = [k]
            if k == 'ma02ha01':
                to_known = ['ma02', 'ha01']

            step1_est = np.loadtxt(to_twostep + d + k + 'Known/step1/coefs.csv')
            step2_est = np.loadtxt(to_twostep + d + k + 'Known/optimum.csv')
            step1_data = pd.read_csv(to_twostep + d + k + 'Known/step1/step1_data.csv')
            cities_to_keep = get_cities_to_keep(step1_data)

            cov = twostep_cov(step1_est,
                              step2_est,
                              step1_data,
                              cities_to_keep,
                              h = 1.0e-05,
                              cities_to_drop = to_drop,
                              cities_to_known = to_known)

            np.savetxt(to_twostep + d + k + 'Known/twostep_var.csv', cov)



## Testing
path = './results/ancient/twostep/noneDrop/base/'
#
### Test 1: I get back the standard errors of stage 2
#step2_est = np.loadtxt(path + 'optimum.csv')
#
#e = estimate.EstimateAncient('directional')
#jac = e.get_jacobian(step2_est, zeta_fixed = True)
#step2_errors = e.get_errors(step2_est)
#g_22 = G_22(jac)
#g_22_inv = np.linalg.inv(g_22)
#h2 = get_h2(jac, step2_errors)
#s_22 = S_22(h2)
#cov = np.linalg.multi_dot((g_22_inv, s_22, np.transpose(g_22_inv)))
#cov_other = np.loadtxt(path + 'cov_white.csv', delimiter=',')
##np.testing.assert_almost_equal((1.0 / len(jac)) * cov, cov_other)
#
#
### Test 2: I get back the standard errors of stage 1
#step1_est = np.loadtxt(path + 'step1/coefs.csv')
#step1_data = pd.read_csv(path + 'step1/step1_data.csv')
#y = step1_data['s_ij'].values
#X = step1_data.drop(['id_i', 'id_j', 's_ij'], axis=1).values
#h1 = get_h1(step1_est, y, X)
#g_11 = G_11(step1_est, y, X)
#g_11_inv = np.linalg.inv(g_11)
#s_11 = S_11(h1)
##cov = (1.0/len(step1_data)) * np.linalg.multi_dot((g_11_inv, s_11, np.transpose(g_11_inv)))
#cov = np.linalg.multi_dot((g_11_inv, s_11, np.transpose(g_11_inv)))
#cov_other = np.loadtxt(path + 'step1/var.csv', delimiter=',')
#np.testing.assert_almost_equal(cov, cov_other)


# Test 3: these standard errors should be larger than the white s.e. (for
# everysing except possibly zeta)
twostep_var = np.loadtxt(path + 'twostep_var.csv')

step1_est = np.loadtxt(path + 'step1/coefs.csv')
step2_est = np.loadtxt(path + 'optimum.csv')
step1_data = pd.read_csv(path + 'step1/step1_data.csv')
cities_to_keep = get_cities_to_keep(step1_data)
twostep_c = twostep_cov(step1_est,
                          step2_est,
                          step1_data,
                          cities_to_keep,
                          h = 1.0e-05)
twostep_var = np.diag(twostep_c)
cov = np.loadtxt(path + 'cov_white.csv', delimiter=',')
var = np.diag(cov)
comp = np.column_stack((twostep_var, np.concatenate(([0.0], var))))
