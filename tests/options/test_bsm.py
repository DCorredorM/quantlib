import unittest
from quantlib.options import *


class TestBlackScholes(unittest.TestCase):
	def test_call_formula(self):
		option = CallOption(underlying_ticker='TSLA', strike_price=800, maturity=1)
		vol = HistoricalVolatility('TSLA')

		model = bsm.BlackScholes(option_contract=option, volatility_model=vol)
		model.value()


if __name__ == '__main__':
	unittest.main()
