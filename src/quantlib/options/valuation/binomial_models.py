import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
from typing import overload

from quantlib.options.options import OptionContract, OptionTypes
from .valuation_abstract import ValuationModel
from utils.miscellaneous import color_map


class BinomialModel(nx.DiGraph, ValuationModel):
    """docstring for Binomial_Model"""
    # TODO: Make it depend of a general payoff function!
    def __init__(self,
                 option_contract: OptionContract,
                 up_factor: float,
                 down_factor: float,
                 time_delta: float,
                 risk_free_rate: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.option_contract = option_contract
        self.underlying_price = option_contract.underlying_price
        self.strike_price = option_contract.strike_price
        self.maturity = option_contract.maturity

        self.up_factor = up_factor
        self.down_factor = down_factor
        self.time_delta = time_delta
        self.risk_free_rate = risk_free_rate
        self.risk_neutral_probability = (1 + self.risk_free_rate - self.down_factor) / \
                                        (self.up_factor - self.down_factor)

        self.create_binomial_tree()
        self.root = [i for i in self.nodes if self.nodes[i]["period"] == 0][0]
        self.visualizer = BinomialTreeVisualizer(self)
        self.from_tree = False

    @classmethod
    def from_tree(cls,
                  price_tree: nx.DiGraph,
                  option_contract: OptionContract,
                  risk_free_rate: float,
                  **kwargs):
        """
        Builds a BinomialModel from a given tree (n.xDiGraph object).

        Parameters
        ----------
        price_tree
        option_contract
        risk_free_rate
        kwargs

        Returns
        -------

        """
        obj = cls.__new__(cls)

        # Calls the init of the nx.DiGraph class with the incoming tree.
        super(cls, obj).__init__(price_tree, **kwargs)

        # Fills the rest of the needed arguments.
        obj.option_contract = option_contract
        obj.maturity = option_contract.maturity
        obj.underlying_price = option_contract.underlying_price
        obj.strike_price = option_contract.strike_price
        obj.risk_free_rate = risk_free_rate
        obj.visualizer = BinomialTreeVisualizer(obj)
        obj.from_tree = True
        obj.root = [i for i in obj.nodes if obj.nodes[i]["period"] == 0][0]
        return obj

    def create_binomial_tree(self):
        self.add_node((0, 0), price=self.underlying_price, period=0, ups=0)
        self.create_sons((0, 0))

    def create_sons(self, node):
        if self.nodes[node]["period"] + self.time_delta <= self.maturity:
            n1 = (node[0] + 1, node[1] + 1)
            if n1 not in self.nodes:
                self.add_node(n1,
                              price=self.nodes[node]["price"] * self.up_factor,
                              period=self.nodes[node]["period"] + self.time_delta,
                              ups=self.nodes[node]["ups"] + 1)
                self.create_sons(n1)
            self.add_edge(node, n1)

            n2 = (node[0] + 1, node[1])
            if n2 not in self.nodes:
                self.add_node(n2,
                              price=self.nodes[node]["price"] * self.down_factor,
                              period=self.nodes[node]["period"] + self.time_delta,
                              ups=self.nodes[node]["ups"])
                self.create_sons(n2)
            self.add_edge(node, n2)

    def value_synthetic_asset_portfolio(self):
        """
        Computes the value of the option finding the synthetics asset price..
        Returns
        -------
        float
            The value of the underlying option.
        """

        def value(node):
            n = self.nodes[node]
            if 'value' not in n.keys():
                if np.isclose(n["period"], self.maturity):
                    n["delta"] = 0
                    n["beta"] = 0
                    n["value"] = self.option_contract.payoff(n['price'])
                else:
                    n_down, n_up = tuple(sorted(list(self.successors(node)), key=lambda x: self.nodes[x]['period']))
                    vd = value(n_down)
                    vu = value(n_up)
                    sd = self.nodes[n_down]['price']
                    su = self.nodes[n_up]['price']

                    n["delta"] = (vu - vd) / (su - sd)
                    n["beta"] = (su * vd - sd * vu) / ((1 + self.risk_free_rate) * (su - sd))
                    n["value"] = n["price"] * n["delta"] + n["beta"]
            return n["value"]

        return value(self.root)

    def value(self):
        if self.option_contract.theoretical_price is None:
            self.option_contract.theoretical_price = self.value_synthetic_asset_portfolio()
        return self.value_synthetic_asset_portfolio()

    def calc_rebal_prom(self):
        self.value_synthetic_asset_portfolio()
        mean_rebal = 0
        for i, j in self.edges():
            self[i][j]["re_balace"] = abs(self.nodes[i]["delta"] - self.nodes[j]["delta"])
            mean_rebal += self[i][j]["re_balace"]
        return mean_rebal / len(self.edges())


class BinomialTreeVisualizer:
    def __init__(self, tree: BinomialModel):
        self.tree = tree

    def plot_price_tree(self):
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = {i: (self.tree.nodes[i]["period"], self.tree.nodes[i]["price"]) for i in self.tree.nodes()}
        labels = {i: round(self.tree.nodes[i]["price"], 2) for i in self.tree.nodes()}

        nx.draw_networkx(self.tree, pos=pos, node_size=1700, node_color="lavender", alpha=0.7, font_size=8, linewidths=1,
                         edgecolors="black", labels=labels, ax=ax)
        plt.show()

    def plot_val_tree(self, delta=True):
        self.tree.value_synthetic_asset_portfolio()
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = {i: (self.tree.nodes[i]["period"], self.tree.nodes[i]["price"]) for i in self.tree.nodes()}
        cmap = color_map(0, max([self.tree.nodes[i]["value"] for i in self.tree.nodes]), True)
        node_color = [cmap(self.tree.nodes[i]["value"]) for i in self.tree.nodes]

        labels = {i: 'V: ' + str(round(self.tree.nodes[i]["value"], 2)) + '\n $\Delta: $' + str(
            round(self.tree.nodes[i]["delta"], 2)) for i in self.tree.nodes()}
        nx.draw_networkx(self.tree, pos=pos, node_size=1600, node_color=node_color, alpha=0.3, linewidths=1,
                         edgecolors="black", node_shape="o", ax=ax, with_labels=False)
        nx.draw_networkx_labels(self.tree, pos, labels, font_size=8)

        plt.show()