from .valuation_abstract import *
from quantlib.options import *
from quantlib.volatility import *
from quantlib.data_handler import *

from quantlib.utils.miscellaneous import to_date

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
		if self.option_contract.theoretical_price is None:
			value, greeks = BlackScholes.bs_call(
				underlying_price=self.underlying_price,
				risk_free_rate=self.risk_free_rate(),
				volatility=self.volatility_model(),
				strike_price=self.option_contract.strike_price,
				maturity=self.option_contract.maturity)
			self.greeks = greeks
			if self.option_contract.type == OptionTypes.call:
				self.option_contract.theoretical_price = value
			elif self.option_contract.type == OptionTypes.put:
				self.option_contract.theoretical_price = \
					BlackScholes.put_call_parity_call_to_put(call_price=value,
					                                         strike_price=self.option_contract.strike_price,
					                                         risk_free_rate=self.risk_free_rate(),
					                                         underlying_price=self.underlying_price,
					                                         maturity=self.option_contract.maturity)
			else:
				raise AttributeError(f'Black-Sholes can only value call and put european options. '
				                     f'Use a binomial model to value your option.')
		return self.option_contract.theoretical_price

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

		# Compute the ITM probability
		itm_probability = sts.norm.cdf(d2)

		# Delta (dV / dS)
		delta = sts.norm.cdf(d1)

		# Compute the BS price
		c = s_0 * delta - K * np.exp(-r * T) * itm_probability

		# Gamma (dV^2 / dS^2)
		gamma = 1 / (s_0 * sigma * np.sqrt(2 * np.pi * T)) * np.exp(- (d1 ** 2 / 2))

		# Theta (dV / dT)
		theta = - ((s_0 * sigma) / (2 * np.sqrt(2 * np.pi * T))) * np.exp(- (d1 ** 2 / 2)) \
		        - r * K * np.exp(- r * T) * itm_probability

		# Vega (dV / d sigma)
		vega = (s_0 * np.sqrt(T) / np.sqrt(2 * np.pi)) * np.exp(- (d1 ** 2 / 2))

		# Rho (dV / dr)
		rho = T * K * np.exp(-r * T) * itm_probability

		return c, Greeks(delta, gamma, theta, vega, rho, itm_probability)

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


class MonteCarloBlackScholes(ValuationModel):
	# Todo: Implement it!
	def value(self):
		pass


class Greeks:
	# Todo: Improve the design of this class...
	def __init__(self,
	             delta,
	             gamma,
	             theta,
	             vega,
	             rho,
	             itm_probability):
		self.delta = delta
		self.gamma = gamma
		self.theta = theta
		self.vega = vega
		self.rho = rho
		self.itm_probability = itm_probability

		self.table = pd.DataFrame(
			{'Greeks': ['Delta (dV / dS)', 'Gamma (dV^2 / dS^2)', 'Theta (dV / dT)',
			            'Vega (dV / dsigma)', 'Rho (dV / dr)', 'ITM probability'],
			 'Value': [self.delta, self.gamma, self.theta, self.vega, self.rho, self.itm_probability]})

	def __repr__(self):
		return self.table.to_markdown()

