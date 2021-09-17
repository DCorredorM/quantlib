from abc import ABC, abstractmethod
from quantlib.options.options import OptionContract, OptionTypes


class ValuationModel(ABC):
	def __init__(self, option_contract: OptionContract):
		self.option_contract = option_contract

	@abstractmethod
	def value(self):
		...
