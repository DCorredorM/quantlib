from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Union

from datetime import datetime


class OptionTypes:
	call = 0
	put = 1
	exotic = 2


class PayOffType:
	european = 0
	american = 1
	asian = 2


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
	             strike_price: float,
	             maturity:Union[float, datetime],
	             underlying_ticker: str = None,
	             payoff_type: int = PayOffType.european):

		self._strike_price = strike_price
		self._maturity = maturity
		self.underlying_ticker = underlying_ticker

		self.valuation_model = None
		self.type = None
		self.theoretical_price = None
		self.payoff_type = payoff_type



	def _check_operation(self, other):
		pass_ = True
		if isinstance(other, OptionContract):
			if other.payoff_type == self.payoff_type:
				if other.maturity == self.maturity:
					return pass_
				else:
					raise Exception(f'Is not possible to operate contracts with different payoffs yet.')
			else:
				raise Exception(f'Is not possible to operate contracts with different payoff types.')

		else:
			raise Exception(f'Is not possible to subtract objects of type OptionContract and {type(other)}')

	def __add__(self, other):
		if self._check_operation(other):
			def payoff(x):
				return self.payoff(x) + other.payoff(x)
			exotic = ExoticOption(payoff_function=payoff, maturity=self.maturity,payoff_type=self.payoff_type)
			return exotic

	def __sub__(self, other):
		if self._check_operation(other):
			def payoff(x):
				return self.payoff(x) - other.payoff(x)
			exotic = ExoticOption(payoff_function=payoff, maturity=self.maturity,payoff_type=self.payoff_type)
			return exotic

	def __mul__(self, other):
		if isinstance(other, float) or isinstance(other, int):
			def payoff(x):
				return self.payoff(x) * other
			exotic = ExoticOption(payoff_function=payoff, maturity=self.maturity,payoff_type=self.payoff_type)
			return exotic
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
	             maturity:Union[float, datetime],
	             underlying_ticker: str = None,
	             payoff_type: int = PayOffType.european):

		super().__init__(strike_price, maturity, underlying_ticker, payoff_type)
		self.type = OptionTypes.call

	def payoff(self, underlying_price):
		return np.maximum(underlying_price - self.strike_price, 0)


class PutOption(OptionContract):
	def __init__(self,
	             strike_price: float,
	             maturity:Union[float, datetime],
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

		return self.payoff_function(underlying_price, *args,  **kwargs)
