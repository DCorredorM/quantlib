import unittest
from quantlib.data_handler import *


class TestDataHandler(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.data = self.test_data_import()

	def test_data_import(self):
		tickers = ['SPY', 'KO', 'NIO']
		data = Data(tickers, start_date=date(2018, 9, 4), end_date=date(2021, 9, 2))
		assert len(data._raw_data['NIO']) > 0
		return data

	def test_add_indicator(self):
		self.data.add_ta_indicator(indicator_name='RSIIndicator',
		                           indicator_kwargs={'window': 17})

	def test_add_indicator_2(self):
		initial_cols = len(self.data._raw_data['SPY'].columns)
		self.data.add_ta_indicator(indicator_name='StochasticOscillator',
		                           indicator_kwargs={'window': 14,
		                                             'smooth_window': 3,
		                                             'fillna': False})
		assert len(self.data._raw_data['SPY'].columns) == initial_cols + 2

	def test_candle_chart(self):
		self.data.visualizer.candle_chart()


if __name__ == '__main__':
	unittest.main()
