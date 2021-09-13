import pandas as pd
from pandas_datareader import data as data_reader
from datetime import date, timedelta
import numpy as np
from copy import copy
import os
from ta.volume import *
from ta.volatility import *
from ta.others import *
from ta.trend import *
from ta.momentum import *
from ta import add_all_ta_features

import yfinance as yf

from utils.miscellaneous import *


class Data:
	engine = 'yahoo'

	def __init__(self, tickers, start_date=None, end_date=None, shift=252, **kwargs):
		if Data.engine == 'iex':
			if kwargs.get('iex_api_key') is None and "IEX_API_KEY" not in os.environ.keys():
				raise Exception('If you want to use iex you need an api key. '
				                'Please create an account here https://iexcloud.io/ and provide'
				                'a valid api key under the argument "iex_api_key".')
			os.environ["IEX_API_KEY"] = kwargs.get('iex_api_key', os.environ["IEX_API_KEY"])
		elif Data.engine == 'yahoo':
			yf.pdr_override()

		if start_date is None:
			start_date = date.today() - timedelta(days=shift)
			end_date = date.today()
		self.tickers = tickers
		
		# Pulls data
		self.raw_data = self.get_data(start_date, end_date)
		
		# Creates log returns dataframe
		self.log_returns = pd.DataFrame({k: self.logarithmic_returns(self.raw_data[k], 1) for k in self.tickers},
		                                columns=self.tickers)
		self.expected_returns = self.log_returns.mean(axis=0)
		# Adds log returns to raw data dataframes:
		for i in self.tickers:
			log_ret = self.log_returns[i].rename('log_return')
			self.raw_data[i] = pd.concat([self.raw_data[i], log_ret], axis=1)
		# Covariance matrix
		self.covariance = self.log_returns.cov()
		
		# Additional indicators dictionary
		self.indicators = {}		

	def get_data(self, start_date, end_date):
		data = {}
		for i in copy(self.tickers):
			try:
				if Data.engine == 'yahoo':
					data[i] = data_reader.get_data_yahoo(i, data_source='yahoo', start=start_date, end=end_date)
				else:
					data[i] = data_reader.DataReader(i, data_source=Data.engine, start=start_date, end=end_date)
				data[i] = data[i].rename(columns={i: i.lower() for i in data[i].columns})
			except Exception as e:
				print(f"There was an error retrieving data for {i}, removing the ticker from the ticker list...\n{e}")
				self.tickers.remove(i)
		return data

	@staticmethod
	def logarithmic_returns(data, shift):
		column = Data._select_close_column(list(data.columns))
		log_return = np.log(data[column]) - np.log(data[column].shift(shift))
		log_return = log_return.dropna(how='all')
		return log_return

	@staticmethod
	def _select_close_column(columns):
		cols = list(map(lambda x: x.replace(' ', '').lower(), columns))
		if 'adjclose' in cols:
			return columns[cols.index('adjclose')]
		elif 'close' in cols:
			return columns[cols.index('close')]
		else:
			raise Exception('No close column in datasets.')

	def add_ta_indicator(self, indicator_name, indicator_kwargs):
		self.indicators[indicator_name] = {}
		for ticker in self.tickers:
			for k in ['open', 'high', 'low', 'close', 'volume']:
				if k in indicator_kwargs.keys():
					exec(f'{k}_ = self.raw_data[ticker]["{k}"]')
					indicator_kwargs[k] = f'{k}_'
			self.indicators[indicator_name][ticker] = eval(f'{indicator_name}({", ".join([f"{key}={value}" for key, value in indicator_kwargs.items()])})')
			if 'Indicator' in indicator_name:
				method_name = from_camel_case_to_underscore(indicator_name.replace('Indicator', ''))
				indicator = eval(f'self.indicators[indicator_name][ticker].{method_name}()')
				if indicator.name not in self.raw_data[ticker].columns:
					self.raw_data[ticker] = pd.concat([self.raw_data[ticker], indicator], axis=1)

	def add_all_indicators(self):
		for ticker in self.tickers:
			self.raw_data[ticker] = add_all_ta_features(self.raw_data[ticker], open="open", high="high", low="low", close="close", volume="volume", fillna=True)

