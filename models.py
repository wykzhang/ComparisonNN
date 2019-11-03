from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt


def LSTM(x_train, y_train, x_test, y_test, shuffle=False, batch_size=64):
    model_type = "LSTM"
    model_saver = ModelCheckpoint(filepath='/models/{}.hdf5'.format(model_type),
                                  monitor='val_loss', save_best_only=True, save_weights_only=False,
                                  mode='auto', save_freq='epoch')
    model_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0)
    regressor = Sequential()
    regressor.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.4))
    regressor.add(LSTM(units=256, return_sequences=False))
    regressor.add(Dropout(0.3))
    regressor.add((Dense(units=256)))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units=y_train.shape[1]))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

    history = regressor.fit(x=x_train, y=y_train, epochs=100, batch_size=batch_size,
                            validation_data=(x_test, y_test), shuffle=shuffle,
                            callbacks=[model_saver, model_stopper],
                            verbose=2
                            )
    # regressor.save('models/{}.h5'.format(model_type))
    return regressor, history


def FFNN(x_train, y_train, x_test, y_test, shuffle=False, batch_size=64):
    model_type = "FFNN"
    model_saver = ModelCheckpoint(filepath='/models/{}.hdf5'.format(model_type),
                                  monitor='val_loss', save_best_only=True, save_weights_only=False,
                                  mode='auto', save_freq='epoch')
    model_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0)

    regressor = Sequential()
    regressor.add(Dense(units=256))
    regressor.add(Dropout(0.4))
    regressor.add(Dense(units=256))
    regressor.add(Dropout(0.3))
    regressor.add((Dense(units=256)))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units=y_train.shape[1]))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

    history = regressor.fit(x=x_train, y=y_train, epochs=100, batch_size=batch_size,
                            validation_data=(x_test, y_test), shuffle=shuffle,
                            callbacks=[model_saver, model_stopper],
                            verbose=2
                            )
    # regressor.save('models/{}.h5'.format(model_type))
    return regressor, history


def RNN(x_train, y_train, x_test, y_test, shuffle=False, batch_size=64):
    model_type = "RNN"
    model_saver = ModelCheckpoint(filepath='/models/{}.hdf5'.format(model_type),
                                  monitor='val_loss', save_best_only=True, save_weights_only=False,
                                  mode='auto', save_freq='epoch')
    model_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=3, verbose=0)

    regressor = Sequential()
    regressor.add(SimpleRNN(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.4))
    regressor.add(SimpleRNN(units=256, return_sequences=False))
    regressor.add(Dropout(0.3))
    regressor.add((Dense(units=256)))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units=y_train.shape[1]))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])

    history = regressor.fit(x=x_train, y=y_train, epochs=100, batch_size=batch_size,
                            validation_data=(x_test, y_test), shuffle=shuffle,
                            callbacks=[model_saver, model_stopper],
                            verbose=2
                            )
    # regressor.save('models/{}.h5'.format(model_type))
    return regressor, history


def generate_images(history, model_type):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}_Accuracy.png'.format(model_type))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}_Loss.png'.format(model_type))


if __name__ == '__main__':
    print('ok')
    model_type = "To_be_determined"

