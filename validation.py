from tensorflow.keras.models import load_model
from data import process_data

#method used to comute return based on each model.

if __name__ == "__main__":
    model_type = "LSTM"
    model = load_model('models/{}.h5'.format(model_type))

    x_train, y_train, x_test, y_test = process_data(time_series=True, debug=True)
    total_return = 0
    for x in range(x_test.shape[0]):
        y_pred = model.predict(x_test[x:x+1])
        for y in range(y_pred.shape[1]):
            if y_pred[0, y] > 0:
                total_return += (100* y_test[x, y])
            else:
                total_return += (-100* y_test[x, y])

    print(total_return/x_test.shape[0])

    print('ok')