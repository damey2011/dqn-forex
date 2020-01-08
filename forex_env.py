import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

STATE_RANGE = 30
INPUT_DATA_COL_INDEX = [0, 1, 2, 3, 4]
CLOSING_PRICE_INDEX = 3
TIME_JUMP = 2


class ForexEnv:
    def __init__(self, pair='EURUSD', balance=1000, lot=1.0, is_test=True, auto_reset_env=True, train_data=True):
        self.pair = pair
        self.balance = balance
        self.is_test = is_test
        """0: buy, 1: sell, 2: do nothing"""
        self.action_space = [
            0, 1, 2
        ]
        self.action_space_n = len(self.action_space)
        self.state_space_n = STATE_RANGE * len(INPUT_DATA_COL_INDEX)
        self.lot = lot
        self.open_position_exists = False
        self.sl = -50
        self.tp = 150
        """
        Current Assumptions:
            1. We are using a fixed lot size
            2. Maximum draw down of 20 pips
            3. We are using the close prices only right now, we might want to include the highs and the lows later so
                the agent can know where it's wrong in cases of hitting sl/drawdown
            4. The market doesnt reach both the SL and the TP in same period
        """
        """Test Params:"""
        self.pointer = STATE_RANGE
        self.current_position = 'buy'
        self.current_trade_highest = None
        self.current_trade_lowest = None
        self.entry_price = None
        self.entry_pointer_index = None
        self.current_profit = 0
        self.auto_reset_env = auto_reset_env
        self.train_data = train_data

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if is_test:
            self.data = self.load_data()

    def divide_data(self):
        df2 = pd.read_csv(
            './data/EURUSD_Candlestick_4_Hour_BID_01.01.2010-04.01.2020.csv',
            sep=',',
            low_memory=False,
            header=None
        )
        df2.columns = ['Local Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        rows = df2.shape[0]
        train_rows = int(0.70 * rows)
        test_rows = rows - train_rows
        df2.head(train_rows).to_csv('./data/EURUSD_4H_TRAIN.csv', index=None, header=None)
        df2.tail(test_rows).to_csv('./data/EURUSD_4H_TEST.csv', index=None, header=None)

    def load_data(self):
        if self.train_data:
            df2 = pd.read_csv(
                './data/EURUSD_4H_TRAIN.csv',
                sep=',',
                low_memory=False,
            )
        else:
            df2 = pd.read_csv(
                './data/EURUSD_4H_TEST.csv',
                sep=',',
                low_memory=False,
            )
        df2.columns = ['Local Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df2[['Volume']] = self.scaler.fit_transform(df2[['Volume']])
        df2 = df2.drop(['Local Time'], axis=1)
        return df2.to_numpy()

    def get_pair_mult_index(self):
        dz_per_pip = 10000
        if self.pair == 'EURUSD':
            dz_per_pip = 10000
        return dz_per_pip

    def calculate_pips(self, start, target):
        # Start is always entry price
        if self.current_position == 'buy':
            profit = target - start
        else:
            profit = start - target
        return round(profit * self.get_pair_mult_index(), 2)

    def calculate_balance(self, pips):
        if self.pair == 'EURUSD':
            # It's $10 per pip for EURUSD
            self.balance += (pips * self.lot * 10)

    def current_state(self):
        start = self.pointer - STATE_RANGE
        state = list(self.data[start: self.pointer, INPUT_DATA_COL_INDEX].flatten())
        return state

    def get_next_state(self):
        next_pointer = self.pointer + TIME_JUMP
        start = next_pointer - STATE_RANGE
        state = list(self.data[start: next_pointer, INPUT_DATA_COL_INDEX].flatten())
        return state

    def get_current_price(self):
        return self.data[self.pointer][CLOSING_PRICE_INDEX]

    def get_current_date(self):
        return self.data[self.pointer][0]

    def data_range(self, start_index, end_index):
        data_to_check = self.data[start_index: end_index]
        return data_to_check

    @property
    def current_trade_peak_and_bottom(self):
        if self.open_position_exists:
            data_to_check = self.data_range(self.entry_pointer_index, self.pointer)
            return np.max(data_to_check[:, [0, 1, 2, 3]]), np.min(data_to_check[:, [0, 1, 2, 3]])

    def validate_current_trade(self):
        if self.open_position_exists:

            assert self.current_position in ['buy', 'sell'], 'Invalid Position, should either be a buy or a sell'

            # data_to_check = self.data_range(self.entry_pointer_index, self.pointer + 1)
            # if data_to_check.any():
            #     info, pips = 'active', 0
            #     for d in data_to_check:
            #         if self.current_position == 'buy':
            #             temp = d[[0, 1, 2, 3]] - self.entry_price
            #             self.current_trade_lowest = np.min(d[[0, 1, 2, 3]])
            #             self.current_trade_highest = np.max(d[[0, 1, 2, 3]])
            #         else:
            #             temp = self.entry_price - d[[0, 1, 2, 3]]
            #             self.current_trade_lowest = np.max(d[[0, 1, 2, 3]])
            #             self.current_trade_highest = np.min(d[[0, 1, 2, 3]])
            #         lowest = np.min(temp)
            #         highest = np.max(temp)
            #         if lowest * self.get_pair_mult_index() <= self.sl:
            #             info, pips = 'sl_hit', self.sl
            #             break
            #         elif highest * self.get_pair_mult_index() >= self.tp:
            #             info, pips = 'tp_hit', self.tp
            #             break
            #     return info, pips
            # else:
            #     return 'active', 0

            data_to_check = self.data_range(self.entry_pointer_index, self.pointer)
            if data_to_check.any():
                peak, bottom = self.current_trade_peak_and_bottom
                max_pips = self.calculate_pips(self.entry_price, peak)
                min_pips = self.calculate_pips(self.entry_price, bottom)
                if min_pips <= self.sl or max_pips <= self.sl:
                    return 'sl_hit', self.sl
                elif max_pips >= self.tp or min_pips >= self.tp:
                    return 'tp_hit', self.tp
                else:
                    return 'active', 0
            else:
                return 'active', 0

    def tick(self):
        """This fast-forwards time by specified time jump all the time"""
        self.pointer += TIME_JUMP
        # if self.open_position_exists and self.pointer % 15 == 0:
        #     current_price = self.get_current_price()
        # print(f'Entry Price: {self.entry_price}')
        # print(f'Current Price: {current_price}')
        # Only print current profit and prices at intervals of 15
        # profit = self.calculate_pips(self.entry_price, current_price)
        # print(f'Current Profit: {profit} pips \n')

    def execute_test_trade(self, action):
        self.entry_pointer_index = self.pointer
        self.entry_price = self.get_current_price()
        if action == 0:
            self.current_position = 'buy'
            print(f'Entered a buy position at {self.entry_price}')
        elif action == 1:
            self.current_position = 'sell'
            print(f'Entered a sell position at {self.entry_price}')

    def current_trade_reward(self):
        return self.calculate_pips(self.entry_price, self.get_current_price())

    def execute_live_trade(self, action):
        raise NotImplementedError

    def step(self, action):
        self.tick()
        """Returns 'next_state', 'reward', 'done', 'info'"""
        if not self.open_position_exists and action != 2:
            self.open_position_exists = True
            if self.is_test:
                self.execute_test_trade(action)
            else:
                self.execute_live_trade(action)

        if self.open_position_exists:
            outcome, points = self.validate_current_trade()
            self.calculate_balance(points)
            next_state = self.get_next_state()
            if outcome == 'sl_hit':
                if self.auto_reset_env:
                    self.reset()
                return next_state, points, True, 'sl_hit'
            elif outcome == 'tp_hit':
                if self.auto_reset_env:
                    self.reset()
                return next_state, points, True, 'tp_hit'
            else:
                return next_state, 0, False, 'active'

        if not self.open_position_exists and action == 2:
            return self.get_next_state(), -1, True, 'no_trade'

    def reset(self):
        self.open_position_exists = False
        self.entry_price = None
        self.entry_pointer_index = None
        return self.current_state()
