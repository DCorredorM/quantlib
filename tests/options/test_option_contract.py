import unittest
from quantlib.options import *


class TestOptionContract(unittest.TestCase):
    def test_combo(self):
        c = CallOption(underlying_ticker='AAPL', strike_price=140, maturity=1)
        p = PutOption(underlying_ticker='AAPL', strike_price=140, maturity=1)

        combo = c - 2 * p

        combo.visualizer.payoff()
        plt.show()
        
    def test_combo_sum(self):
        c = CallOption(underlying_ticker='AAPL', strike_price=140, maturity=1)
        p = PutOption(underlying_ticker='AAPL', strike_price=140, maturity=1)
    
        combo = sum([c, 2 * p])
    
        combo.visualizer.payoff()
        plt.show()


if __name__ == '__main__':
    unittest.main()
