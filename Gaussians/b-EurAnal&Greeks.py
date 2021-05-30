'''
 Valuation of European call-put options
 in Black-Scholes-Merton model,
 implied volatility estimation
 and all the Greeks
 '''


def bsm_call_value(S0, K, T, r, sigma,  option='call'):
    ''' Valuation of European call option in BSM model.
    Analytical formula.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    value: float
        present value of the European call option
    '''
    from math import log, sqrt, exp
    from scipy import stats

    #print(stats.norm.cdf(40, 4, 30))
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    # stats.norm.cdf --> cumulative distribution function
    #                    for normal distribution
    if option == 'call':
        value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) -
                K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    else:
        value = (-S0 * stats.norm.cdf(-d1, 0.0, 1.0) +
                K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0))
    return value


def bsm_vega(S0, K, T, r, sigma):
    ''' Vega of European option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    vega: float
        partial derivative of BSM formula with respect
        to sigma, i.e. Vega

    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega

# Implied volatility function


def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    ''' Implied volatility of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma_est: float
        estimate of impl. volatility
    it: integer
        number of iterations

    Returns
    =======
    simga_est: float
        numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0) /
                      bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est

#print ("bsm_call_value(100, 105, 1, 0.02, 0.2): = $", bsm_call_value(100, 105, 1, 0.02, 0.2))
print(bsm_call_imp_vol(100, 105, 1, 0.02, bsm_call_value(100, 105, 1, 0.02, 0.2), 0.5))
# since delta for c is N(d1)

def bsm_delta(S0, K, T, r, sigma, option ='call'):
    ''' Delta of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    vega: float
        partial derivative of BSM formula with respect
        to S, i.e. Delta

    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option=='call':
        return stats.norm.cdf(d1, 0.0, 1.0)
    return stats.norm.cdf(d1, 0.0, 1.0) - 1

def bsm_gamma(S0, K, T, r, sigma):
    ''' Delta of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    vega: float
        partial derivative of BSM formula with respect
        to S, i.e. Delta

    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    gamma = stats.norm.cdf(d1, 0.0, 1.0) / (S0*sigma*sqrt(T))
    return gamma

def bsm_rho(S0, K, T, r, sigma, option = 'call'):
    ''' Delta of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    rho: float
        partial derivative of BSM formula with respect
        to r, i.e. rho

    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    #d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option == 'call':
        return K*T*stats.norm.cdf(d2)
    return -K*T*stats.norm.cdf(-d2)

def bsm_theta(S0, K, T, r, sigma, option = 'call'):
    ''' Delta of European call option in BSM model.

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity date (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Returns
    =======
    rho: float
        partial derivative of BSM formula with respect
        to r, i.e. rho

    '''
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option == 'call':
        return -S0*stats.norm.pdf(d1)*sigma/(2*sqrt(T))-r*K*exp(-r*T)*stats.norm.cdf(d2)
    return -S0*stats.norm.pdf(d1)*sigma/(2*sqrt(T))+r*K*exp(-r*T)*stats.norm.cdf(-d2)


# Facciamo un grafico analitico delta - c as a function of K
import numpy as np

ec = []
dec = []
vec =[]
gec =[]
tec =[]
rec =[]

ep = []
dep = []
vep =[]
gep =[]
tep =[]
rep =[]

k_list = np.arange(80., 120.1, 5.)
for K in k_list:
    ec.append(bsm_call_value(100, K, 1, 0.02, 0.2))
    dec.append(bsm_delta(100, K, 1, 0.02, 0.2))
    vec.append(bsm_vega(100, K, 1, 0.02, 0.2))
    gec.append(bsm_gamma(100, K, 1, 0.02, 0.2))
    tec.append(bsm_theta(100, K, 1, 0.02, 0.2))
    rec.append(bsm_rho(100, K, 1, 0.02, 0.2))
    ep.append(bsm_call_value(100, K, 1, 0.02, 0.2, option = 'put'))
    dep.append(bsm_delta(100, K, 1, 0.02, 0.2, option = 'put'))
    tep.append(bsm_theta(100, K, 1, 0.02, 0.2, option = 'put'))
    rep.append(bsm_rho(100, K, 1, 0.02, 0.2, option = 'put'))

ec = np.array(ec)
dec = np.array(dec)
vec = np.array(vec)
gec = np.array(gec)
tec = np.array(tec)
rec = np.array(rec)


from pylab import plt
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, figsize=(15, 10))
ax1.plot(k_list, ec, 'b', label='call ')
ax1.plot(k_list, ep, 'r', label='put ')
ax1.set_ylabel('Eur Opt value')
fig.legend(loc=1)
ax2.plot(k_list, dec, 'b', label='Delta')
ax2.plot(k_list, dep, 'r', label='Delta')
ax2.set_ylabel('Delta')
ax5.plot(k_list, tec, 'b', label='Tetha')
ax5.plot(k_list, tep, 'r', label='Tetha')
ax5.set_ylabel('Theta')
ax6.plot(k_list, rec, 'b', label='Rho')
ax6.plot(k_list, rep, 'r', label='Rho')
ax6.set_ylabel('Rho')
ax3.plot(k_list, vec, 'c', label='Vega')
ax3.set_ylabel('Vega')
ax4.plot(k_list, gec, 'c', label='Gamma')
ax4.set_ylabel('Gamma')
ax5.set_xlabel('strike')
ax5.set_xlim(left=75, right=125)
ax6.set_xlabel('strike')
ax6.set_xlim(left=75, right=125)
plt.savefig('Fig/b-GreeksEurOp')
plt.show()