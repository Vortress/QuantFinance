import numpy as np
from scipy.stats import norm
from dataclasses import dataclass


@dataclass
class Equity:
    spot: float
    dividend_yield: float
    volatility: float

@dataclass
class EquityOption:
    strike : float
    time_to_maturity : float
    put_call : str

@dataclass
class EquityForward:
    strike: float
    time_to_maturity: float

def fwd_pricer(underlying, option, rate):
    return underlying.spot*np.exp(-underlying.dividend_yield*option.time_to_maturity) - option.strike*np.exp(-rate*option.time_to_maturity)


def bsm_pricer(underlying, option, rate):
    d1 = (np.log(underlying.spot / option.strike) + option.time_to_maturity * (
                rate - underlying.dividend_yield + 0.5 * underlying.volatility ** 2)) / (
                     underlying.volatility * np.sqrt(option.time_to_maturity))
    d2 = d1 - underlying.volatility * np.sqrt(option.time_to_maturity)
    if option.put_call == "call":
        return underlying.spot * np.exp(-underlying.dividend_yield * option.time_to_maturity) * norm.cdf(
            d1) - option.strike * np.exp(-rate * option.time_to_maturity) * norm.cdf(d2)
    if option.put_call == "put":
        return -underlying.spot * np.exp(-underlying.dividend_yield * option.time_to_maturity) * norm.cdf(
            -d1) + option.strike * np.exp(-rate * option.time_to_maturity) * norm.cdf(-d2)


def bsm_delta(underlying, option, rate):
    spot2 = underlying.spot + 0.001
    underlying2 = Equity(spot2, underlying.dividend_yield, underlying.volatility)
    return (bsm_pricer(underlying2, option, rate) - bsm_pricer(underlying, option, rate))/0.001

def bsm_gamma(underlying, option, rate):
    spot2 = underlying.spot + 0.001
    underlying2 = Equity(spot2, underlying.dividend_yield, underlying.volatility)
    spot3 = underlying.spot - 0.001
    underlying3 = Equity(spot3, underlying.dividend_yield, underlying.volatility)
    return (bsm_pricer(underlying2, option, rate) + bsm_pricer(underlying3, option, rate) - 2*bsm_pricer(underlying, option, rate))/(0.001**2) 


