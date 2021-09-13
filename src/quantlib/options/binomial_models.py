import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Binomial_Model(nx.DiGraph):
	"""docstring for Binomial_Model"""
	# TODO: Make it depend of a general payoff function!

	def __init__(self, spot_price, up_factor, down_factor, maturity, time_delta, strike_price, risk_free_rate):
		super(Binomial_Model, self).__init__()
		self.spot_price = spot_price
		self.up_factor = up_factor
		self.down_factor = down_factor
		self.maturity = maturity
		self.time_delta = time_delta
		self.strike_price = strike_price
		self.risk_free_rate = risk_free_rate
		self.create_binomial_tree()

	def create_binomial_tree(self):
		# print("nodes0",len(self.nodes))
		# self.create_sons(0)
		self.add_node(0, p=self.spot_price, per=0, ups=0)
		nt = [0]
		for t in range(1, self.maturity + 1):
			oldnt = nt
			nt = []
			for i in range(0, t + 1):
				nt += [len(self.nodes)]
				self.add_node(len(self.nodes), p=self.spot_price * self.up_factor ** i * self.down_factor ** (t - i), per=t, ups=i)
			for i in nt:
				for j in oldnt:
					# print(self.nodes[j]["p"],self.nodes[i]["p"],self.nodes[j]["ups"]-self.nodes[i]["ups"])
					d = self.nodes[i]["ups"] - self.nodes[j]["ups"]
					if d == 1 or d == 0:
						self.add_edge(j, i)

	def create_sons(self, node):
		if self.nodes[node]["per"] + self.time_delta <= self.maturity:
			n1 = len(self.nodes)

			self.add_node(n1, p=self.nodes[node]["p"] + self.up_factor, per=self.nodes[node]["per"] + self.time_delta)
			self.create_sons(n1)

			n2 = len(self.nodes)
			self.add_node(n2, p=self.nodes[node]["p"] - self.down_factor, per=self.nodes[node]["per"] + self.time_delta)
			self.create_sons(n2)

	def plot_price_tree(self):
		pos = {i: (self.nodes[i]["per"], (self.nodes[i]["ups"] - self.nodes[i]["per"] / 2)) for i in self.nodes()}
		labels = {i: round(self.nodes[i]["p"], 2) for i in self.nodes()}

		nx.draw_networkx(self, pos=pos, node_size=1700, node_color="beige", alpha=1, font_size=8, linewidths=1,
		                 edgecolors="black", labels=labels)
		plt.show()

	def calc_val_RP(self):
		pi = (1 + self.risk_free_rate - self.down_factor) / (self.up_factor - self.down_factor)
		for t in list(range(self.maturity + 1))[::-1]:
			for i in self.nodes():
				if self.nodes[i]["per"] == t:
					if t == self.maturity:
						self.nodes[i]["val"] = max(0, self.nodes[i]["p"] - self.strike_price)
						self.nodes[i]["delta"] = 0
					else:
						cu = max(self.nodes[k]["val"] for k in self.successors(i))
						cd = min(self.nodes[k]["val"] for k in self.successors(i))
						su = max(self.nodes[k]["p"] for k in self.successors(i))
						sd = min(self.nodes[k]["p"] for k in self.successors(i))
						self.nodes[i]["delta"] = (cu - cd) / (su - sd)
						self.nodes[i]["val"] = (1 / (1 + r)) * (pi * cu + (1 - pi) * cd)
		return self.nodes[0]["val"]

	def calc_val_Port(self):
		pi = (1 + self.risk_free_rate - self.down_factor) / (self.up_factor - self.down_factor)
		print(pi)
		print(1 - pi)
		for t in list(range(self.maturity + 1))[::-1]:
			for i in self.nodes():
				if self.nodes[i]["per"] == t:
					if t == self.maturity:
						self.nodes[i]["val"] = max(0, self.nodes[i]["p"] - self.strike_price)
						self.nodes[i]["delta"] = 0
					else:
						cu = max(self.nodes[k]["val"] for k in self.successors(i))
						cd = min(self.nodes[k]["val"] for k in self.successors(i))
						su = max(self.nodes[k]["p"] for k in self.successors(i))
						sd = min(self.nodes[k]["p"] for k in self.successors(i))
						self.nodes[i]["delta"] = (cu - cd) / (su - sd)
						self.nodes[i]["beta"] = (su * cd - sd * cu) / ((1 + self.risk_free_rate) * (su - sd))
						self.nodes[i]["val"] = self.nodes[i]["p"] * self.nodes[i]["delta"] + self.nodes[i]["beta"]
		return self.nodes[0]["val"]

	def plot_val_tree(self, delta=True):
		pos = {i: (self.nodes[i]["per"], (self.nodes[i]["ups"] - self.nodes[i]["per"] / 2)) for i in self.nodes()}

		if delta:
			labels = {i: 'C: ' + str(round(self.nodes[i]["val"], 2)) + '\n $\Delta: $' + str(
				round(self.nodes[i]["delta"], 2)) for i in self.nodes()}
			nx.draw_networkx(self, pos=pos, node_size=1600, node_color="beige", alpha=1, font_size=8, linewidths=1,
			                 edgecolors="black", labels=labels, node_shape="o")
		else:
			labels = {i: round(self.nodes[i]["val"], 2) for i in self.nodes()}
			nx.draw_networkx(self, pos=pos, node_size=1000, node_color="beige", alpha=1, font_size=8, linewidths=1,
			                 edgecolors="black", labels=labels)

		plt.show()

	def calc_rebal_prom(self):
		self.calc_val_Port()
		mean_rebal = 0
		for i, j in self.edges():
			self[i][j]["re_balace"] = abs(self.nodes[i]["delta"] - self.nodes[j]["delta"])
			mean_rebal += self[i][j]["re_balace"]
		return mean_rebal / len(self.edges())


if __name__ == '__main__':
	S0 = 150

	T = 10
	dt = 0.01
	K = 115
	sigma = 0.2
	rf = 0.02
	u = np.exp(sigma * np.sqrt(dt / T))
	d = 1 / u
	r = np.exp(rf * dt / T) - 1

	print(u)
	print(r + 1)
	print(d)

	BT = Binomial_Model(S0, u, d, T, dt, K, r)
	print(len(BT.nodes()))
	# BT.plot_price_tree()

	print(BT.calc_val_Port())
	BT.plot_val_tree()
	BT.plot_price_tree()

	print("promedio delta", BT.calc_rebal_prom())
