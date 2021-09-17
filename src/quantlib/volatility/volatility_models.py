from abc import abstractmethod, ABC
from quantlib.data_handler.source import Data

import datetime

import numpy as np


class VolatilityModel(ABC):
	def __init__(self, ticker):
		self.ticker = ticker
		pass

	def __call__(self, *args, **kwargs):
		return self._calculate_volatility(*args, **kwargs)

	@abstractmethod
	def _calculate_volatility(self, *args, **kwargs):
		...


class HistoricalVolatility(VolatilityModel):
	def __init__(self, ticker, data: Data = None):
		super().__init__(ticker)
		if data is None:
			data = Data(tickers=ticker, shift=252 * 4)
		self.data = data

	def _calculate_volatility(self, *args, **kwargs):
		return np.sqrt(self.data.covariance.loc[self.ticker, self.ticker] * 252)


class ConstantVolatility(VolatilityModel):
	def __init__(self, constant):
		super().__init__(None)
		self.constant = constant

	def _calculate_volatility(self, *args, **kwargs):
		return self.constant


class ImpliedVolatility(VolatilityModel):
	def __init__(self, ticker):
		super().__init__(ticker)

	def _calculate_volatility(self, date):
		return self._get_implied_vol(date)

	def _get_implied_vol(self, date: datetime.datetime = None):
		vol = None
		tick = self.ticker
		if date is None:
			date = datetime.datetime.now()
		# TODO: Fetch implied volatility for a given date and ticker.

		return vol


class GarchVolatility(VolatilityModel):
	def __init__(self, ticker, data: Data = None):
		super().__init__(ticker)
		if data is None:
			data = Data(tickers=ticker, shift=252 * 4)
		self.data = data

	def _calculate_volatility(self, date):
		...


class ArchVolatility(VolatilityModel):
	def __init__(self, ticker, data: Data):
		super().__init__(ticker)
		if data is None:
			data = Data(tickers=ticker, shift=252 * 4)
		self.data = data

	def _calculate_volatility(self, date):
		...
