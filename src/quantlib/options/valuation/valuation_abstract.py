from abc import ABC, abstractmethod


class ValuationModel(ABC):
	@abstractmethod
	def value(self):
		...
