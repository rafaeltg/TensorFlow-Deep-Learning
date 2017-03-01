import json
from sklearn.preprocessing import StandardScaler

from pydl.utils.datasets import *
from pydl.models import RNN
from pydl.model_selection import mse, TimeSeriesCV, RegressionCV
import matplotlib.pyplot as plt

np.random.seed(42)

sp500 = get_stock_historical_data('^GSPC', '2000-01-01', '2017-03-01', True, ['Close'])
look_back = 20
look_ahead = 1
time_steps = 1


def run_forecast1():

    sp500_log_ret = get_log_return(sp500)

    # split into train and test sets
    train_size = int(len(sp500_log_ret) * 0.9)
    train, test = sp500_log_ret[0:train_size], sp500_log_ret[train_size:len(sp500)]
    print(len(train))
    print(len(test))

    pp = StandardScaler()
    train = pp.fit_transform(train)
    test = pp.transform(test)

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train, y_train = create_dataset(train, look_back, look_ahead)
    x_test, y_test = create_dataset(test, look_back, look_ahead)

    x_train = np.reshape(x_train, (x_train.shape[0], time_steps, look_back))
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, look_back))

    print('train = ', len(x_train))
    print('test = ', len(x_test))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    print('Training LSTM')
    lstm.fit(x_train, y_train)

    y_test_pred = lstm.predict(x_test)

    y_test = pp.inverse_transform(y_test)
    y_test_pred = pp.inverse_transform(y_test_pred)

    print(y_test.shape)
    print(y_test_pred.shape)
    print('Test MSE = %.4f' % mse(y_test, y_test_pred))
    print('Test corr = %.4f' % np.corrcoef(y_test[:, 0], y_test_pred[:, 0])[0, 1])

    errs = y_test[:, 0] - y_test_pred[:, 0]
    mean_err = np.average(errs)
    sd_err = np.std(errs)
    print('Test ME = %.4f' % mean_err)
    print('Test SdE = %.4f' % sd_err)

    plt.figure(1)
    #plt.plot(range(len(errs)), errs, label='Errors')
    #plt.axhline(y=mean_err, color='red', linestyle='-')
    #plt.axhline(y=mean_err+sd_err, color='green', linestyle='-')
    #plt.axhline(y=mean_err-sd_err, color='green', linestyle='-')
    plt.plot(range(len(y_test[:, 0])), y_test[:, 0], label='Actual')
    plt.plot(range(len(y_test_pred[:, 0])), y_test_pred[:, 0], label='Pred')
    plt.legend(loc='best')
    plt.show()


def run_forecast2():
    x, y = create_dataset(sp500, look_back, look_ahead)

    # split into train and test sets
    train_size = int(len(x) * 0.85)
    x_train, y_train = x[0:train_size], y[0:train_size]
    x_test, y_test = x[train_size:len(x)], y[train_size:len(y)]

    pp_x = StandardScaler()
    x_train = pp_x.fit_transform(x_train)
    x_test = pp_x.transform(x_test)

    pp_y = StandardScaler()
    y_train = pp_y.fit_transform(y_train)
    y_test = pp_y.transform(y_test)

    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x_train = np.reshape(x_train, (x_train.shape[0], time_steps, look_back))
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, look_back))

    print('train = ', len(x_train))
    print('test = ', len(x_test))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    print('Training LSTM')
    lstm.fit(x_train, y_train)
    y_test_pred = lstm.predict(x_test)

    y_test_pred = pp_y.inverse_transform(y_test_pred)
    y_test = pp_y.inverse_transform(y_test)

    test_pred = y_test_pred[range(0, len(y_test_pred), look_ahead)]
    test_pred = np.reshape(test_pred, (y_test.shape[0], 1))

    print(y_test.shape)
    print(test_pred.shape)
    print('Test MSE = %.3f' % mse(y_test.flatten(), test_pred.flatten()))
    print('Test corr = %.3f' % np.corrcoef(y_test.flatten(), test_pred.flatten())[0, 1])

    errs = y_test - test_pred
    mean_err = np.average(errs)
    sd_err = np.std(errs)
    print('Test ME = %.3f' % mean_err)
    print('Test SdE = %.3f' % sd_err)

    '''
    plt.figure(1)
    plt.plot(range(len(errs)), errs, label='Errors')
    plt.axhline(y=mean_err, color='red', linestyle='-')
    plt.axhline(y=mean_err+sd_err, color='green', linestyle='-')
    plt.axhline(y=mean_err-sd_err, color='green', linestyle='-')
    #plt.plot(range(len(test_pred)), test_pred, label='Pred')
    #plt.plot(range(len(test)), test, label='Actual')
    plt.legend(loc='best')
    plt.show()
    '''


def run_cv():

    print(len(sp500))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    cv = TimeSeriesCV(window=1334, horizon=365, by=365, fixed=True, dataset_fn=create_dataset)

    print('Running CV!')
    res = cv.run(model=lstm, ts=sp500, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))


def _create_dataset(x):
    # reshape into X=[t-look_back, t] and Y=[t+1, t+look_ahead]
    x, y = create_dataset(x, look_back, look_ahead)
    x = np.reshape(x, (x.shape[0], time_steps, look_back))
    return x, y


def run_cv1():

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    x, y = create_dataset(sp500, look_back, look_ahead)
    print(len(x))

    cv = RegressionCV(method='time_series', window=1319, horizon=365, by=365, fixed=True)

    print('Running CV!')
    res = cv.run(model=lstm, x=x, y=y, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))


def test():

    sp500_log_ret = get_log_return(sp500)
    x, y = create_dataset(sp500_log_ret, look_back, look_ahead)
    print(len(x))

    lstm = RNN(cell_type='lstm',
               layers=[50, 50],
               stateful=False,
               time_steps=time_steps,
               num_epochs=300,
               batch_size=200,
               opt='adam')

    cv = RegressionCV(method='time_series', window=1512, horizon=252, by=252, fixed=True)

    print('Running CV!')
    res = cv.run(model=lstm, x=x, y=y, metrics=['mape', 'rmse'], max_thread=2)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))


if __name__ == '__main__':
    #test()
    #run_cv()
    #run_cv1()
    run_forecast1()
    #run_forecast2()