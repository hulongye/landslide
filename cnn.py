from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras import regularizers


def cnn():
    model = Sequential()
    conv1 = Conv1D(filters=15, kernel_size=3, padding='same', activation='tanh', input_shape=(13, 1),
                   kernel_regularizer=regularizers.L1(0.01))
    pool1 = MaxPooling1D(strides=1)
    conv2 = Conv1D(filters=30, kernel_size=3, padding='same', activation='tanh')
    pool2 = MaxPooling1D(strides=1)
    conv3 = Conv1D(filters=60, kernel_size=3, padding='same', activation='tanh')
    pool3 = MaxPooling1D(strides=1)
    model.add(conv1)
    model.add(pool1)
    model.add(conv2)
    model.add(pool2)
    model.add(conv3)
    model.add(pool3)
    model.add(Flatten())
    model.add(Dense(60, activation='tanh', kernel_regularizer=regularizers.L1(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model