import pandas as pd
from pandas_datareader import data as data_reader
from datetime import date, timedelta
import numpy as np
from copy import copy
import os
import logging


from ta import *
from ta.volume import *
from ta.volatility import *
from ta.trend import *
from ta.momentum import *

from ta.utils import IndicatorMixin


import inspect

import yfinance as yf

from quantlib.utils.miscellaneous import *
from typing import Union, List

from plotly import graph_objs as go

import requests
from bs4 import BeautifulSoup


class Data:
    engine = 'yahoo'

    def __init__(self, tickers: Union[str, List[str]], start_date=None, end_date=None, shift=252, **kwargs):
        if Data.engine == 'iex':
            if kwargs.get('iex_api_key') is None and "IEX_API_KEY" not in os.environ.keys():
                raise Exception('If you want to use iex you need an api key. '
                                'Please create an account here https://iexcloud.io/ and provide'
                                'a valid api key under the argument "iex_api_key".')
            os.environ["IEX_API_KEY"] = kwargs.get('iex_api_key', os.environ["IEX_API_KEY"])
        elif Data.engine == 'yahoo':
            yf.pdr_override()

        if end_date is None:
            end_date = date.today()

        if start_date is None:
            start_date = end_date - timedelta(days=shift)

        if isinstance(tickers, list):
            self.tickers = tickers
        elif isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            raise AttributeError('The tickers need to be either a list or a string')

        # Pulls data
        self._raw_data = self.get_data(start_date, end_date)
        self.raw_data = self._raw_data

        # Creates log returns dataframe
        self.log_returns = pd.DataFrame({k: self.logarithmic_returns(self._raw_data[k], 1) for k in self.tickers},
                                        columns=self.tickers)
        self.expected_returns = self.log_returns.mean(axis=0)
        # Adds log returns to raw data dataframes:
        for i in self.tickers:
            log_ret = self.log_returns[i].rename('log_return')
            self._raw_data[i] = pd.concat([self._raw_data[i], log_ret], axis=1)
        # Covariance matrix
        self.covariance = self.log_returns.cov()

        # Additional indicators dictionary
        self.indicators = {}

        # Visualizer
        self.visualizer = DataVisualizer(self)

    @classmethod
    def from_data(cls, data, tickers):
        obj = cls.__new__(cls)
        if isinstance(tickers, list):
            obj.tickers = tickers
        elif isinstance(tickers, str):
            obj.tickers = [tickers]
            data = {tickers: data}
        else:
            raise AttributeError('The tickers need to be either a list or a string')

        obj._raw_data = data
        obj.raw_data = obj._raw_data

        # Creates log returns dataframe
        obj.log_returns = pd.DataFrame({k: obj.logarithmic_returns(obj._raw_data[k], 1) for k in obj.tickers},
                                        columns=obj.tickers)
        obj.expected_returns = obj.log_returns.mean(axis=0)
        # Adds log returns to raw data dataframes:
        for i in obj.tickers:
            log_ret = obj.log_returns[i].rename('log_return')
            obj._raw_data[i] = pd.concat([obj._raw_data[i], log_ret], axis=1)
        # Covariance matrix
        obj.covariance = obj.log_returns.cov()

        # Additional indicators dictionary
        obj.indicators = {}

        # Visualizer
        obj.visualizer = DataVisualizer(obj)
        return obj

    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data

    @raw_data.getter
    def raw_data(self):
        if len(self.tickers) == 1:
            return self._raw_data[self.tickers[0]]
        else:
            return self._raw_data

    def get_data(self, start_date, end_date):
        data = {}
        for i in copy(self.tickers):
            try:
                if Data.engine == 'yahoo':
                    data[i] = data_reader.get_data_yahoo(i, start=start_date, end=end_date)
                else:
                    data[i] = data_reader.DataReader(i, data_source=Data.engine, start=start_date, end=end_date)
                data[i] = data[i].rename(columns={i: i.lower() for i in data[i].columns})
            except Exception as e:
                print(f"There was an error retrieving data for {i}, removing the ticker from the ticker list...\n{e}")
                self.tickers.remove(i)
        return data

    @staticmethod
    def logarithmic_returns(data, shift):
        column = Data._select_close_column(list(data.columns))
        log_return = np.log(data[column]) - np.log(data[column].shift(shift))
        log_return = log_return.dropna(how='all')
        return log_return

    @staticmethod
    def _select_close_column(columns):
        cols = list(map(lambda x: x.replace(' ', '').lower(), columns))
        if 'adjclose' in cols:
            return columns[cols.index('adjclose')]
        elif 'close' in cols:
            return columns[cols.index('close')]
        else:
            raise Exception('No close column in datasets.')

    def add_ta_indicator(self, indicator_name, indicator_kwargs):
        """

        Parameters
        ----------
        indicator_name
        indicator_kwargs

        Returns
        -------

        """
        self.indicators[indicator_name] = {}
        req_params = list(inspect.signature(eval(indicator_name)).parameters)
        for ticker in self.tickers:
            for k in ['open', 'high', 'low', 'close', 'volume']:
                if k in req_params:
                    exec(f'{k}_ = self.raw_data[ticker]["{k}"]')
                    indicator_kwargs[k] = f'{k}_'
            self.indicators[indicator_name][ticker] = eval(
                f'{indicator_name}({", ".join([f"{key}={value}" for key, value in indicator_kwargs.items()])})'
            )
            indicator_df = Data._append_indicator(self.indicators[indicator_name][ticker])
            self._raw_data[ticker] = pd.concat([self._raw_data[ticker], indicator_df], axis=1)
            self._raw_data[ticker] = self._raw_data[ticker].loc[:, ~self._raw_data[ticker].columns.duplicated()]

    @staticmethod
    def _append_indicator(indicator: IndicatorMixin) -> pd.DataFrame:
        cols = [i for i in dir(indicator) if '_' != i[0] and eval(f'callable({i})')]
        locs = locals()
        columns = [eval(f'indicator.{i}()', locs) for i in cols]
        return pd.concat(columns, axis=1)

    def add_all_indicators(self):
        for ticker in self.tickers:
            self._raw_data[ticker] = add_all_ta_features(self._raw_data[ticker],
                                                         open="open",
                                                         high="high",
                                                         low="low",
                                                         close="close",
                                                         volume="volume",
                                                         fillna=True)


class DataVisualizer:
    def __init__(self, data: Data):
        self.data = data

    def candle_chart(self, ticker=None):
        ticker = self.data.tickers[0] if ticker is None else ticker
        if ticker not in self.data.tickers:
            raise Exception('The ticker is not in your data')

        if len(self.data.tickers) == 1:
            df = self.data.raw_data
        else:
            df = self.data.raw_data[ticker]

        ohlc = go.Ohlc(x=list(df.index),
                       open=list(df['open']),
                       high=list(df['high']),
                       low=list(df['low']),
                       close=list(df['close']))
        fig = go.Figure(data=ohlc)
        fig.show()


class RiskFreeRateScraper:
    supported_maturities = [0, 0.25, 1, 2, 3, 6, 12]

    @property
    def maturity(self):
        return self._maturity

    @maturity.setter
    def maturity(self, maturity):
        self._maturity = min(RiskFreeRateScraper.supported_maturities, key=lambda x: abs(x - maturity))

    def __init__(self, maturity: int = None, access_date: date = None):
        self._maturity = 12 if maturity is None else maturity
        self.access_date = date.today() if access_date is None else access_date
        self.rates = dict()

    def __call__(self, maturity=None, access_date=None):
        if access_date is None:
            access_date = self.access_date
        if maturity is not None:
            self.maturity = maturity
        rates = self._get_rates_maturity()
        rates.drop_duplicates(inplace=True)
        rates.dropna(axis=0, inplace=True)
        date_ = min(rates.index, key=lambda x: abs(to_date(x) - to_date(access_date)))

        return rates['rate'].loc[date_]

    def _get_rates_maturity(self):
        if self.maturity not in self.rates.keys():
            self.rates[self.maturity] = self._get_rates_maturity_scraping()
        return self.rates[self.maturity]

    def _get_rates_maturity_scraping(self):
        mat = 'overnight' if self.maturity == 0 else '1-week' if self.maturity == 0.25 else \
                f'{self.maturity}-month' if self.maturity == 1 else f'{self.maturity}-months'
        url = f'https://www.global-rates.com/en/interest-rates/libor/american-dollar/usd-libor-interest-rate-{mat}.aspx'

        page = requests.post(url, {'submit': 'btn_usd'})
        soup = BeautifulSoup(page.content, "html.parser")

        html_tables = soup.find_all(cellpadding=2, limit=3)
        rates = None
        for tab in html_tables:
            table = RiskFreeRateScraper.tableDataText(tab)
            if rates is None:
                rates = RiskFreeRateScraper.table_to_df(table)
            else:
                rates = pd.concat([rates, RiskFreeRateScraper.table_to_df(table)])
        return rates

    @staticmethod
    def tableDataText(table):
        """Parses a html segment started with tag <table> followed
        by multiple <tr> (table rows) and inner <td> (table data) tags. 
        It returns a list of rows with inner columns. 
        Accepts only one <th> (table header/data) in the first row.
        """

        def rowgetDataText(tr, coltag='td'):  # td (data) or th (header)
            return [td.get_text(strip=True) for td in tr.find_all(coltag)]

        rows = []
        trs = table.find_all('tr')
        headerow = rowgetDataText(trs[0], 'th')
        if headerow:  # if there is a header row include first
            rows.append(headerow)
            trs = trs[1:]
        for tr in trs:  # for every table row
            rows.append(rowgetDataText(tr, 'td'))  # data row
        return rows

    @staticmethod
    def table_to_df(table):
        df = {'date': [], 'rate': []}
        for row in table[1:]:
            df['date'].append(string_date_to_datetime(row[0]))
            if row[1] == '-':
                df['rate'].append(np.nan)
            else:
                df['rate'].append(float(row[1].replace('\xa0%', '')) / 100)

        df = pd.DataFrame(df)
        df.set_index('date', inplace=True)
        return df
