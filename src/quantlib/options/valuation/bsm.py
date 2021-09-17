from .valuation_abstract import *
from quantlib.options import *
from quantlib.volatility import *
from quantlib.data_handler import *

from utils.miscellaneous import to_date

from typing import Union
import scipy.stats as sts
from datetime import date


class BlackScholes(ValuationModel):
	def __init__(self,
	             option_contract: OptionContract,
	             volatility_model: Union[float, VolatilityModel] = None,
	             risk_free_rate: Union[float, RiskFreeRateScraper] = None,
	             underlying_price: float = None,
	             valuation_date: datetime = None):

		super().__init__(option_contract)
		self.underlying_ticker = option_contract.underlying_ticker

		# Valuation date
		if valuation_date is None:
			valuation_date = date.today()
		elif isinstance(valuation_date, datetime.datetime):
			valuation_date = valuation_date.date()
		self.valuation_date = valuation_date

		# Underlying spot price:
		if underlying_price is None:
			data = Data(option_contract.underlying_ticker, end_date=valuation_date, shift=10)
			self.valuation_date = min(data.raw_data.index,
			                          key=lambda x: abs(valuation_date - to_date(x)))
			self.underlying_price = data.raw_data['close'].loc[self.valuation_date]
		else:
			self.underlying_price = underlying_price

		# Volatility
		if volatility_model is None:
			self.volatility_model = ImpliedVolatility(option_contract.underlying_ticker)
		elif isinstance(volatility_model, VolatilityModel):
			self.volatility_model = volatility_model
		elif isinstance(volatility_model, float):
			self.volatility_model = ConstantVolatility(volatility_model)
		else:
			raise Exception('The volatility_model needs to be one of float or VolatilityModel')

		# Volatility
		if risk_free_rate is None:
			self.risk_free_rate = RiskFreeRateScraper(access_date=self.valuation_date)
		elif isinstance(risk_free_rate, RiskFreeRateScraper):
			self.risk_free_rate = risk_free_rate
		elif isinstance(risk_free_rate, float):
			def r(*args):
				return risk_free_rate

			self.risk_free_rate = r
		else:
			raise Exception('The risk_free_rate needs to be one of float or RiskFreeRateScraper')

	def value(self):
		value = BlackScholes.bs_call(
			underlying_price=self.underlying_price,
			risk_free_rate=self.risk_free_rate(),
			volatility=self.volatility_model(),
			strike_price=self.option_contract.strike_price,
			maturity=self.option_contract.maturity)
		if self.option_contract.type == OptionTypes.call:
			return value
		elif self.option_contract.type == OptionTypes.put:
			return BlackScholes.put_call_parity_call_to_put(call_price=value,
			                                                strike_price=self.option_contract.strike_price,
			                                                risk_free_rate=self.risk_free_rate(),
			                                                underlying_price=self.underlying_price,
			                                                maturity=self.option_contract.maturity)
		else:
			raise AttributeError(f'Black-Sholes can only value call and put european options. '
			                     f'Use a binomial model to value your option.')

	@staticmethod
	def bs_call(underlying_price: float,
	            risk_free_rate: float,
	            volatility: float,
	            strike_price: float,
	            maturity: float):
		"""

		Parameters
		----------
		underlying_price
		risk_free_rate
		volatility
		strike_price
		maturity

		Returns
		-------

		"""
		s_0, r, sigma, K, T = underlying_price, risk_free_rate, volatility, strike_price, maturity

		d1 = (np.log(s_0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		return s_0 * sts.norm.cdf(d1) - K * np.exp(-r * T) * sts.norm.cdf(d2)

	@staticmethod
	def put_call_parity_call_to_put(call_price,
	                                strike_price,
	                                risk_free_rate,
	                                underlying_price,
	                                maturity):
		"""
		Implements the put call parity for valuing put options.

		Parameters
		----------
		call_price
		strike_price
		risk_free_rate
		underlying_price
		maturity

		Returns
		-------

		"""

		return call_price + strike_price * np.exp(-risk_free_rate * maturity) - underlying_price

	@staticmethod
	def put_call_parity_put_to_call(put_price,
	                                strike_price,
	                                risk_free_rate,
	                                spot_price,
	                                maturity):
		"""
		Implements the put call parity for valuing put options.

		Parameters
		----------
		put_price
		strike_price
		risk_free_rate
		spot_price
		maturity

		Returns
		-------

		"""

		return put_price - strike_price * np.exp(-risk_free_rate * maturity) + spot_price
