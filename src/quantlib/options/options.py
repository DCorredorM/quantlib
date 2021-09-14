from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class OptionTypes:
	call = 0
	put = 1
	american_call = 2
	american_put = 3
	asian = 4
	exotic = 5


class OptionContract(ABC):
	def __init__(self,
	             underlying_price: float,
	             strike_price: float,
	             maturity: int):

		self.underlying_price = underlying_price
		self.strike_price = strike_price
		self.maturity = maturity

		self.valuation_model = None
		self.type = None
		self.theoretical_price = None

	@abstractmethod
	def payoff(self, underlying_price):
		...

	def add_valuation_model(self, valuation_model):
		self.valuation_model = valuation_model(self)


class CallOption(OptionContract):
	def __init__(self,
	             underlying_price: float,
	             strike_price: float,
	             maturity: int):

		super().__init__(underlying_price, strike_price, maturity)
		self.type = OptionTypes.call

	def payoff(self, underlying_price):
		return np.maximum(underlying_price - self.strike_price, 0)


class PutOption(OptionContract):
	def __init__(self,
	             underlying_price: float,
	             strike_price: float,
	             maturity: int):
		super(PutOption).__init__(underlying_price, strike_price, maturity)
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
	             underlying_price: float,
	             strike_price: float,
	             maturity: int,
	             payoff_function: Callable):

		super().__init__(underlying_price, strike_price, maturity)
		self.type = OptionTypes.exotic
		self.payoff_function = payoff_function

	def payoff(self, underlying_price, *args, **kwargs):
		parameters = self.payoff_function.__code__.co_varnames
		if 'strike_price' in parameters:
			kwargs['strike_price'] = self.strike_price
		if 'maturity' in parameters:
			kwargs['maturity'] = self.maturity

		return self.payoff_function(underlying_price, *args,  **kwargs)
