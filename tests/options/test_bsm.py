import unittest
from quantlib.options import *


class TestBlackScholes(unittest.TestCase):
	def test_call_formula(self):
		option = CallOption(underlying_ticker='AAPL', strike_price=140, maturity=0.3)
		# vol = HistoricalVolatility('TSLA')
		vol = ConstantVolatility(0.38)

		model = bsm.BlackScholes(option_contract=option, volatility_model=vol)
		model.value()

		print(model.greeks)


if __name__ == '__main__':
	unittest.main()
