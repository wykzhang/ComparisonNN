import datetime as dt
import pandas_datareader.data as web
import os
import csv
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook
import numpy as np
from sklearn.model_selection import train_test_split


def get_stocks():
    file = open('stock_list.txt', 'r', encoding='utf-8-sig')
    # the file should only have one line with all tickers
    for line in file:
        # SHOULD only iterate once, splits all the tickers into a list to iterate through.
        data = [n for n in line.split(',')]

    folder_name = 'stocks_data'
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()
    for ticker in data:
        # If the file doesn't exist, create the file with the initial start date of 2000
        if not os.path.exists('{}/{}.csv'.format(folder_name, ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('{}/{}.csv'.format(folder_name, ticker))
        else:
            # If the file already exists, check to see if the file needs to be updated.
            with open('{}/{}.csv'.format(folder_name, ticker), 'r') as csvFile:
                reader = csv.reader(csvFile)
                file = [row for row in reader]
                last_date = file[-1][0].split('-')
                last_date = dt.datetime(int(last_date[0]), int(last_date[1]), int(last_date[2]))
                # Compares the last date in the CSV with the current day
                if end.date() != last_date.date():
                    # If the dates are different, move back the last_date to make the call by one day.
                    # This is to ensure that we have correct information when making the call.
                    last_date -= dt.timedelta(days=1)
                    # If the dates are different, grab the data frame starting from the last date in csv
                    df = web.DataReader(ticker, 'yahoo', last_date.date(), end.date())
                    # Convert data frame into a list of strings.
                    to_csv = df.to_csv(header=None).split('\r\n')[1:-1]
                    # This section removes the last row in the existing csv and replaces it with the first row from the df.
                    # This is to overwrite the old value, in case the value was taken before the closing time of that day.
                    file[len(file) - 1] = to_csv[0].split(',')
                    # After that, we iterate over the new values, starting from the 1st index.
                    for line in to_csv[1:]:
                        file.append(line.split(','))
                    with open('{}/{}.csv'.format(folder_name, ticker), 'w', newline='') as writeFile:
                        writer = csv.writer(writeFile)
                        writer.writerows(file)
            csvFile.close()
            writeFile.close()
            print('Updated {}'.format(ticker))


def group_everything():
    """
    Gets all the stocks from stocks_data folder that match the list from stock_list.txt
    Converts it into a single CSV file, with each stock as a column, and each is that stocks closing price for that day
    """
    file = open('stock_list.txt', 'r', encoding='utf-8-sig')
    # the file should only have one line with all tickers
    for line in file:
        # SHOULD only iterate once, splits all the tickers into a list to iterate through.
        tickers = [n for n in line.split(',')]

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stocks_data/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('joined_files/joined_close_stocks.csv')


def trim_split(mat, batch_size, split_percentage=0.8, shuffle=False, time_series=True):
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        mat = mat[no_of_rows_drop - 2:]
    else:
        mat = mat

    train_size = round(mat.shape[0] / batch_size * split_percentage) * batch_size + 1

    mat_train, mat_test = train_test_split(mat, train_size=train_size, shuffle=shuffle)

    if time_series:
        X_train = mat_train[:-1]
        Y_train = mat_train[1:, 0, :]
        X_test = mat_test[:-1]
        Y_test = mat_test[1:, 0, :]
    else:
        X_train = mat_train[:-1]
        Y_train = mat_train[1:]
        X_test = mat_test[:-1]
        Y_test = mat_test[1:]

    # Remove all infinite values.
    X_train[X_train == np.inf] = 0
    Y_train[Y_train == np.inf] = 0
    X_test[X_test == np.inf] = 0
    Y_test[Y_test == np.inf] = 0
    X_train[np.isneginf(X_train)] = 0
    Y_train[np.isneginf(Y_train)] = 0
    X_test[np.isneginf(X_test)] = 0
    Y_test[np.isneginf(Y_test)] = 0

    # This return x_train, y_train, x_test, y_test
    # There is a shift due to the fact that the label is the next day
    return X_train, Y_train, X_test, Y_test


def build_timeseries(mat, time_step=50):
    dim_0 = mat.shape[0] - time_step
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, time_step, dim_1))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:time_step + i]

    print("length of time-series i/o", x.shape)
    return x


def process_data(path='joined_files/joined_close_stocks.csv',
                 batch_size=64, time_step=50,
                 split_percentage=0.8,
                 shuffle=False,
                 time_series=True,
                 debug=False):

    df = pd.read_csv(path)
    # Remove date column
    df1 = df.iloc[:, 1:]
    df1.fillna(0, inplace=True)
    df3 = df1.pct_change()
    df3.replace([np.inf, -np.inf], np.nan)
    df3.fillna(0, inplace=True)
    df4 = df3.iloc[1:, :].values
    if time_series:
        df4 = build_timeseries(df4, time_step)

    x_train, y_train, x_test, y_test = trim_split(df4, batch_size, split_percentage, shuffle, time_series)
    if debug:
        print("x_train: {}, y_train: {}\nx_test:{}, y_test:{}".format(x_train.shape, y_train.shape,
                                                                      x_test.shape, y_test.shape))
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    # group_everything()
    batch_size = 64
    shuffle = False
    x_train, y_train, x_test, y_test = process_data(time_series=True, debug=True, batch_size=batch_size, shuffle=shuffle)
