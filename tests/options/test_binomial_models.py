import unittest
from quantlib.options import *


class TestBinomialModels(unittest.TestCase):
	def test_recursion(self):
		# Create an option contract object
		option = CallOption(
			strike_price=150,  # K
			maturity=1  # 1
		)

		T = 1
		dt = 0.1
		sigma = 0.2
		rf = 0.02
		u = np.exp(sigma * np.sqrt(dt / T))
		d = 1 / u

		b_tree = BinomialModel(
			option_contract=option,  # Option contract object
			up_factor=u,  # The BS approximation up factor
			down_factor=d,  # The BS approximation down factor
			number_of_steps=dt,  # The time delta
			risk_free_rate=np.exp(rf * dt / T) - 1  # Risk free rate
		)
		# b_tree.visualizer.plot_price_tree()
		# print(b_tree.value("synthetic_portfolio"))
		b_tree.visualizer.value_tree()

	def test_from_tree(self):
		# Create a tree:
		g = nx.DiGraph()
		g.add_node('0', p=20)
		g.add_node('01', p=26)
		g.add_node('00', p=17)
		g.add_node('011', p=32)
		g.add_node('010', p=23)
		g.add_node('000', p=15)
		g.add_node('001', p=22)

		for n in g.nodes():
			g.nodes[n]["per"] = len(n) - 1
			g.nodes[n]["ups"] = sum(int(i) for i in n)
			if len(n) < 3:
				g.add_edge(n, f'{n}1')
				g.add_edge(n, f'{n}0')

		bt = BinomialModel.from_tree(price_tree=g,
		                             maturity=2,
		                             spot_price=20,
		                             strike_price=34,
		                             risk_free_rate=0.025)

		bt.price_tree()

	def test_exotic_option(self):
		# Create a tree:
		g = nx.DiGraph()
		g.add_node('0', p=20)
		g.add_node('01', p=26)
		g.add_node('00', p=17)
		g.add_node('011', p=32)
		g.add_node('010', p=23)
		g.add_node('000', p=15)
		g.add_node('001', p=22)

		for n in g.nodes():
			g.nodes[n]["per"] = len(n) - 1
			g.nodes[n]["ups"] = sum(int(i) for i in n)
			if len(n) < 3:
				g.add_edge(n, f'{n}1')
				g.add_edge(n, f'{n}0')

		def payoff_function(underlying_price, strike_price):
			return max(2 * underlying_price - strike_price, 0)

		option = ExoticOption(
			strike_price=34,
			maturity=2,
			payoff_function=payoff_function)
		b_tree = BinomialModel.from_tree(
			price_tree=g,
			option_contract=option,
			risk_free_rate=0.025)
		b_tree.value()

	def test_from_data(self):
		option = CallOption(underlying_ticker='TSLA', strike_price=800, maturity=1)
		vol = HistoricalVolatility('TSLA')

		number_of_steps = 30
		valuation_date = date(2021, 9, 2)
		model = BinomialModel.from_data(option, number_of_steps, vol, valuation_date=valuation_date)

		model.visualizer.sub_value_tree((0, 0), 10)


if __name__ == '__main__':
	# unittest.main()
	t = TestBinomialModels()
	t.test_from_data()