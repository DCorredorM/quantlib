from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union, List, Dict, Tuple

from datetime import datetime

import matplotlib.pyplot as plt


class OptionTypes:
    call = 0
    put = 1
    exotic = 2
    combo = 3

    @staticmethod
    def to_str(type_):
        return ['Call', 'Put', 'Exotic', 'Combo'][type_]


class PayOffType:
    european = 0
    american = 1
    asian = 2

    @staticmethod
    def to_str(type_):
        return ['European', 'American', 'Asian'][type_]


class OptionContract(ABC):
    """
    Attributes
    __________

    strike_price
    maturity
    underlying_ticker
    valuation_model
    type
    theoretical_price
    """

    @property
    def maturity(self):
        return self._maturity

    @maturity.setter
    def maturity(self, maturity):
        self._maturity = maturity
        self.theoretical_price = None

    @property
    def strike_price(self):
        return self._strike_price

    @strike_price.setter
    def strike_price(self, strike_price):
        self._strike_price = strike_price
        self.theoretical_price = None

    def __init__(self,
                 strike_price: Union[float, List[float]],
                 maturity: Union[float, datetime, List[Union[float, datetime]]],
                 underlying_ticker: str = None,
                 payoff_type: int = PayOffType.european):

        self._strike_price = strike_price
        self._maturity = maturity
        self.underlying_ticker = underlying_ticker

        self.valuation_model = None
        self.type = None
        self.theoretical_price = None
        self.payoff_type = payoff_type

        self.visualizer = OptionContractVisualizer(self)

    def _check_operation(self, other):
        pass_ = True
        if isinstance(other, OptionContract):
            if other.payoff_type == self.payoff_type:
                if other.maturity == self.maturity:
                    return pass_
                else:
                    raise Exception(f'Is not possible to operate contracts with different maturities yet.')
            else:
                raise Exception(f'Is not possible to operate contracts with different payoff types.')

        else:
            raise Exception(f'Is not possible to subtract objects of type OptionContract and {type(other)}')

    def __add__(self, other):
        if isinstance(other, ComboStrategy):
            return other + self
        elif self._check_operation(other):
            contracts = [self, other]
            coefficients = [1, 1]

            return ComboStrategy(contracts, coefficients)
        else:
            return self

    def __sub__(self, other):
        if isinstance(other, ComboStrategy):
            return -1 * (other - self)
        elif self._check_operation(other):
            contracts = [self, other]
            coefficients = [1, -1]

            return ComboStrategy(contracts, coefficients)
        elif isinstance(other, int):
            return self

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            contracts = [self]
            coefficients = [other]

            return ComboStrategy(contracts, coefficients)
        else:
            raise Exception(f'Is not possible to multiply objects of type OptionContract and {type(other)}')

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            contracts = [self]
            coefficients = [other]

            return ComboStrategy(contracts, coefficients)
        else:
            raise Exception(f'Is not possible to multiply objects of type OptionContract and {type(other)}')

    @abstractmethod
    def payoff(self, underlying_price):
        ...

    def add_valuation_model(self, valuation_model):
        self.valuation_model = valuation_model(self)

    def get_time_until_expiration(self, date=None, units='days'):
        if isinstance(self.maturity, datetime):
            if date is None:
                date = datetime.now()

            td = (self.maturity - date).days

            if units == 'months':
                td /= 30
            elif units == 'years':
                td /= 360
            return td
        else:
            return self.maturity


class CallOption(OptionContract):
    def __init__(self,
                 strike_price: float,
                 maturity: Union[float, datetime],
                 underlying_ticker: str = None,
                 payoff_type: int = PayOffType.european):
        super().__init__(strike_price, maturity, underlying_ticker, payoff_type)
        self.type = OptionTypes.call

    def payoff(self, underlying_price):
        return np.maximum(underlying_price - self.strike_price, 0)


class PutOption(OptionContract):
    def __init__(self,
                 strike_price: float,
                 maturity: Union[float, datetime],
                 underlying_ticker: str = None,
                 payoff_type: int = PayOffType.european):
        super().__init__(strike_price, maturity, underlying_ticker, payoff_type)
        self.type = OptionTypes.put

    def payoff(self, underlying_price):
        return np.maximum(self.strike_price - underlying_price, 0)


class ExoticOption(OptionContract):
    """
    This ExoticOption object lets the user create an option with an arbitrary payoff function.

    This arbitrary payoff function should be passed as a parameter when constructing the object.
    This function may need objects information such as the strike price or the maturity of the contract, the payoff method
    of the class passes those object parameters to the callable given in the constructors if the names of these
    parameters are as the names of the class attributes, namely strike_price or maturity.

    Parameters
    ----------
    # TODO: Finish docstring...
    underlying_price
    strike_price
    maturity
    payoff_function
    """

    def __init__(self,
                 payoff_function: Callable,
                 maturity: Union[float, datetime],
                 strike_price: float = None,
                 underlying_ticker: str = None,
                 payoff_type: int = PayOffType.european):

        super().__init__(strike_price, maturity, underlying_ticker, payoff_type)
        self.type = OptionTypes.exotic
        self.payoff_function = payoff_function

    def payoff(self, underlying_price, *args, **kwargs):
        parameters = self.payoff_function.__code__.co_varnames
        if 'strike_price' in parameters:
            kwargs['strike_price'] = self.strike_price
        if 'maturity' in parameters:
            kwargs['maturity'] = self.maturity

        return self.payoff_function(underlying_price, *args, **kwargs)


class ComboStrategy(OptionContract):

    def __init__(self, contracts: List[OptionContract], coefficients: List[Union[int]]):

        payoff_type, underlying_ticker = ComboStrategy.check_combo(contracts)
        self.contracts = contracts
        self.coefficients = coefficients

        super(ComboStrategy, self).__init__(
            strike_price=[c.strike_price for c in contracts],
            maturity=[c.maturity for c in contracts],
            underlying_ticker=underlying_ticker,
            payoff_type=payoff_type)

    def __add__(self, other):
        if isinstance(other, ComboStrategy):
            contracts = self.contracts + other.contracts
            coefficients = self.coefficients + other.coefficients

            return ComboStrategy(contracts, coefficients)
        elif isinstance(other, OptionContract):
            contracts = self.contracts + [other]
            coefficients = self.coefficients + [1]
            return ComboStrategy(contracts, coefficients)
        elif isinstance(other, int):
            return self
    
    def __sub__(self, other):
        if isinstance(other, ComboStrategy):
            contracts = self.contracts + other.contracts
            coefficients = self.coefficients + list(map(lambda x: -1 * x, other.coefficients))

            return ComboStrategy(contracts, coefficients)
        elif isinstance(other, OptionContract):
            contracts = self.contracts + [other]
            coefficients = self.coefficients + [-1]
            return ComboStrategy(contracts, coefficients)
        elif isinstance(other, int):
            return self
    
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            contracts = self.contracts
            coefficients = list(map(lambda x: other * x, self.coefficients))

            return ComboStrategy(contracts, coefficients)
        else:
            raise Exception(f'Is not possible to multiply objects of type OptionContract and {type(other)}')

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            contracts = self.contracts
            coefficients = list(map(lambda x: other * x, self.coefficients))

            return ComboStrategy(contracts, coefficients)
        else:
            raise Exception(f'Is not possible to multiply objects of type OptionContract and {type(other)}')

    def payoff(self, underlying_price):
        return sum(c * o.payoff(underlying_price) for c, o in zip(self.coefficients, self.contracts))

    @staticmethod
    def check_combo(contracts):
        payoff_type = contracts[0].payoff_type
        underlying_ticker = contracts[0].underlying_ticker

        for c in contracts[1:]:
            if c.payoff_type != payoff_type:
                raise AttributeError('The contracts conforming a combo strategy need to be of the same payoff type. '
                                     f'You have one of type {PayOffType.to_str(payoff_type)} '
                                     f'and one of type {PayOffType.to_str(c.payoff_type)}')
            elif c.underlying_ticker != underlying_ticker:
                raise AttributeError('The contracts conforming a combo strategy need to have the same underlying.'
                                     f'One underlying is {underlying_ticker}'
                                     f'and other {c.underlying_ticker}')

        return payoff_type, underlying_ticker


class DiagonalComboStrategy(OptionContract):
    def __init__(self, contracts: List[OptionContract], coefficients: List[Union[int, float]]):
        payoff_type, underlying_ticker = ComboStrategy.check_combo(contracts=contracts)
        # For each maturity we create a ComboStrategy object
        self.contracts: Dict[Union[int, datetime], ComboStrategy] = dict()

        for coefficient, contract in zip(coefficients, contracts):
            if contract.maturity in self.contracts:
                self.contracts[contract.maturity] += coefficient * contract
                
            else:
                self.contracts[contract.maturity] = coefficient * contract
         
        super(DiagonalComboStrategy, self).__init__(
            strike_price=[c.strike_price for c in contracts],
            maturity=[c.maturity for c in contracts],
            underlying_ticker=underlying_ticker,
            payoff_type=payoff_type
        )

    def payoff(self, underlying_price: Union[Dict[Union[int, datetime], float], List[float]]):
        if isinstance(underlying_price, Dict):
            return sum(
                self.contracts[maturity].payoff(price) for maturity, price in underlying_price.items()
            )
        else:
            contracts = list(self.contracts.values())
            return sum(contracts[i].payoff(price) for i, price in enumerate(underlying_price))
    
    def __add__(self, other):
        pass


class OptionContractVisualizer:
    def __init__(self, option_contract: OptionContract):
        self.option_contract = option_contract

        if isinstance(self.option_contract, ComboStrategy):
            self.x_low = min([c.strike_price for c in self.option_contract.contracts]) * 0.95
            self.x_high = max([c.strike_price for c in self.option_contract.contracts]) * 1.05
        elif isinstance(self.option_contract, DiagonalComboStrategy):
            self.x_low = min(self.option_contract.strike_price) * 0.95
            self.x_high = max(self.option_contract.strike_price) * 1.05
        else:
            self.x_low = self.option_contract.strike_price * 0.95
            self.x_high = self.option_contract.strike_price * 1.05

    def payoff(self, value=None, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 7)))
        prices = np.linspace(kwargs.pop("x_low", self.x_low), kwargs.pop("x_high",self.x_high), 500)
        
        if value is None:
            y = self.option_contract.payoff(prices)
        else:
            y = self.option_contract.payoff(prices) - value
        ax.plot(prices, y, **kwargs)
        plt.xlabel('Price')
        plt.ylabel('Value')
        return ax
