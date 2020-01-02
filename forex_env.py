import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DAY_MAP = {'Monday': 0.1, 'Tuesday': 0.2, 'Wednesday': 0.3, 'Thursday': 0.4, 'Friday': 0.5, 'Saturday': 0.6,
           'Sunday': 0.7}

STATE_RANGE = 10

USE_TIME_FRAME = 15
# This is only valid in this case when the dataset is in 1min

DATA_RANGE = [4, 5]
DATA_RANGE[1] += 1
# This is the index range of columns we want to include in the state

INPUT_DATA_COL_INDEX = 2

TIME_JUMP = 30

IS_COMPLETE_DATA_STRUCTURE = True


class ForexEnv:
    def __init__(self, pair='EURUSD', lot=1.0, is_test=True, auto_reset_env=True, train_data=True):
        self.pair = pair
        self.is_test = is_test
        """0: buy, 1: sell, 2: do nothing"""
        self.action_space = [
            0, 1, 2
        ]
        self.action_space_n = len(self.action_space)
        self.state_space_n = STATE_RANGE * (DATA_RANGE[1] - DATA_RANGE[0])
        self.lot = lot
        self.open_position_exists = False
        self.sl = -10
        self.tp = 20
        """
        Current Assumptions:
            1. We are using a fixed lot size
            2. Maximum draw down of 20 pips
            3. We are using the close prices only right now, we might want to include the highs and the lows later so
                the agent can know where it's wrong in cases of hitting sl/drawdown
        """
        """Test Params:"""
        self.pointer = STATE_RANGE
        self.current_position = 'buy'
        self.entry_price = None
        self.entry_pointer_index = None
        self.current_profit = 0
        self.auto_reset_env = auto_reset_env
        self.train_data = train_data

        if is_test:
            self.data = self.load_data()

    # def divide_data(self):
    #     df2 = pd.read_csv(
    #         './data/EURUSD-D1-2010-2019.csv',
    #         sep=',',
    #         low_memory=False,
    #         header=None
    #     )
    #     rows = df2.shape[0]
    #     train_rows = int(0.75 * rows)
    #     test_rows = rows - train_rows
    #     df2.head(train_rows).to_csv('./data/EURUSD_2010_2019_TRAIN.csv', index=None, header=None)
    #     df2.tail(test_rows).to_csv('./data/EURUSD_2010_2019_TEST.csv', index=None, header=None)

    def load_data(self, complete=True):
        if self.train_data:
            df2 = pd.read_csv(
                './data/EURUSD_TRAIN.csv',
                sep=',',
                low_memory=False,
            )
        else:
            df2 = pd.read_csv(
                './data/EURUSD_TRAIN.csv',
                sep=',',
                low_memory=False,
            )
        if IS_COMPLETE_DATA_STRUCTURE:
            df2.columns = ['Pair', 'Date', 'Time', 'Open', 'Close', 'Low', 'High', 'Volume']
            df2.Time = df2.Time / 1000000
            df2.Date = pd.to_datetime(df2.Date, format='%Y%m%d')
            df2['WeekDay'] = df2.Date.dt.day_name().map(DAY_MAP)
            df2 = df2.reindex(columns=['WeekDay', 'Time', 'Open', 'Close', 'Low', 'High'])
        else:
            df2.columns = ['Date', 'Close', 'High', 'Low']
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

    def current_state(self):
        RANGE = STATE_RANGE * USE_TIME_FRAME
        start = min(self.pointer - RANGE, 0) if self.pointer < RANGE else self.pointer - RANGE
        # We are considering closing prices as the states, hence why we have index 3
        return list(self.data[start: self.pointer: USE_TIME_FRAME, DATA_RANGE[0]:DATA_RANGE[1]].flatten())

    def get_next_state(self):
        RANGE = STATE_RANGE * USE_TIME_FRAME
        next_pointer = self.pointer + TIME_JUMP
        start = min(next_pointer - RANGE, 0) if next_pointer < RANGE else next_pointer - RANGE
        # We are considering closing prices as the states, hence why we have index 3
        return list(self.data[start: next_pointer: USE_TIME_FRAME, DATA_RANGE[0]:DATA_RANGE[1]].flatten())

    def get_current_price(self):
        return self.data[self.pointer][3]

    def data_range(self, start_index, end_index):
        data_to_check = self.data[start_index: end_index]
        return data_to_check

    @property
    def current_trade_peak_and_bottom(self):
        if self.open_position_exists:
            data_to_check = self.data_range(self.entry_pointer_index, self.pointer)
            if IS_COMPLETE_DATA_STRUCTURE:
                return np.max(data_to_check[:, 4:6]), np.min(data_to_check[:, 4:6])
            return np.max(data_to_check[:, 2:4]), np.min(data_to_check[:, 2:4])

    def validate_current_trade(self):
        if self.open_position_exists:

            assert self.current_position in ['buy', 'sell'], 'Invalid Position, should either be a buy or a sell'

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


#
# env = ForexEnv()
# acc_profits = 0
# acc_losses = 0
# for i in range(10000):
#     done = False
#     env.reset()
#     peak_price = 0
#     profits, losses = 0, 0
#     while not done:
#         action = np.random.choice(env.action_space)
#         next_state, reward, done, info = env.step(action)
#         if done:
#             if info == 'sl_hit':
#                 peak_price = np.min([env.current_trade_peak_and_bottom])
#                 losses -= reward
#             if info == 'tp_hit':
#                 peak_price = np.max([env.current_trade_peak_and_bottom])
#                 profits += reward
#
#     print(f'Got P: {profits}, L: {losses} pips for trade {i}')
#     print(f'Entered at {env.entry_price} and was exited at {peak_price}\n')
#     acc_profits += profits
#     acc_losses -= losses
#
# print(f'Acc profits: {acc_profits}')
# print(f'Acc losses: {acc_losses}')
