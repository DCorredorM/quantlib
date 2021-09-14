from quantlib.data_handler.source import *


def pull_data():
	tickers = ['JNJ', 'AAPL', 'TSLA', 'SPY', 'KO', 'NIO']

	data = Data(tickers, start_date=date(2018, 9, 4), end_date=date(2021, 9, 2))
	print(data.expected_returns)
	print(data.covariance)


if __name__ == '__main__':
	pull_data()