from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam


nb_filters = 32
nb_conv = 3
nb_pool = 2
r_dropout = 0.5


def build(input_shape, n_classes):
    model = Sequential([
        Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=input_shape),
        Activation('relu'),
        Conv2D(nb_filters, (nb_conv, nb_conv)),
        Activation('relu'),
        MaxPooling2D(pool_size=(nb_pool, nb_pool)),
        Activation('relu'),
        MaxPooling2D(),
        Dropout(r_dropout),
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(r_dropout),
        Dense(n_classes),
        Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.summary()

    return model