from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

from datetime import datetime


class OptionTypes:
	call = 0
	put = 1
	american_call = 2
	american_put = 3
	asian = 4
	exotic = 5


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
	def __init__(self,
	             strike_price: float,
	             maturity: Union[int, datetime],
	             underlying_ticker: str = None):

		self.strike_price = strike_price
		self.maturity = maturity
		self.underlying_ticker = underlying_ticker

		self.valuation_model = None
		self.type = None
		self.theoretical_price = None

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
	             maturity: int,
	             underlying_ticker: str = None):

		super().__init__(strike_price, maturity, underlying_ticker)
		self.type = OptionTypes.call

	def payoff(self, underlying_price):
		return np.maximum(underlying_price - self.strike_price, 0)


class PutOption(OptionContract):
	def __init__(self,
	             strike_price: float,
	             maturity: int,
	             underlying_ticker: str = None):
		super(PutOption).__init__(strike_price, maturity, underlying_ticker)
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
	             strike_price: float,
	             maturity: int,
	             payoff_function: Callable,
	             underlying_ticker: str = None):

		super().__init__(strike_price, maturity, underlying_ticker)
		self.type = OptionTypes.exotic
		self.payoff_function = payoff_function

	def payoff(self, underlying_price, *args, **kwargs):
		parameters = self.payoff_function.__code__.co_varnames
		if 'strike_price' in parameters:
			kwargs['strike_price'] = self.strike_price
		if 'maturity' in parameters:
			kwargs['maturity'] = self.maturity

		return self.payoff_function(underlying_price, *args,  **kwargs)
