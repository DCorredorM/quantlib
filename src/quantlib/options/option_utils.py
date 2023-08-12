from fractions import Fraction

from quantlib.options import DiagonalComboStrategy, ComboStrategy

from functools import reduce
import math


def lcm(*values):
    def lcm_2(a, b):
        return abs(a * b) // math.gcd(a, b)
    
    return reduce(lcm_2, values, 1)


def portfolio_multiplier(weights, round_=None):
    if round_:
        fractional_weights_denominators = list(
            map(lambda x: Fraction(str(round(x, round_))).limit_denominator().denominator, weights))
    else:
        fractional_weights_denominators = list(map(lambda x: Fraction(str(x)).limit_denominator().denominator, weights))
    
    return lcm(*fractional_weights_denominators)
