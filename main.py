import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model

import time
import os

FILE_DIR = 'assets/'

file_list = os.listdir(FILE_DIR)

CLASS_LIST = []
text_train = []
text_test = []

split_coefficient = 0.8

for file_name in file_list:
    time.sleep(1)
    m = file_name.split('.')
    class_name = m[0]
    ext = m[1]

    if ext == 'txt':
        if class_name not in CLASS_LIST:
            print(f'new class in list "{class_name}"')
            CLASS_LIST.append(class_name)

        cls = CLASS_LIST.index(class_name)
        print(f'adding file "{file_name}" to class "{CLASS_LIST[cls]}"...')

        with open(f'{FILE_DIR}/{file_name}', 'r', encoding="utf-8") as f:
            text = f.read()
            text = text.replace('\n', ' ').split(' ')
            text_len = len(text)
            text_train.append(' '.join(text[:int(text_len * split_coefficient)]))
            text_test.append(' '.join(text[int(text_len * split_coefficient):]))

CLASS_COUNT = len(CLASS_LIST)

for cls in range(CLASS_COUNT):
    print(f'Class name: {CLASS_LIST[cls]}')
    print(f'fragment from train sample: {text_train[cls][:200]}')
    print(f'fragment from test sample: {text_test[cls][:200]}')
    print('--')


class timex:
    def __enter__(self):
        self.startTime = time.time()
        return self

    def __exit__(self, type, value, traceback):
        print('Время обработки: {:.2f} с'.format(time.time() - self.startTime))


def make_tokenizer(
    VOCAB_SIZE,
    txt_train
):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE,
                          filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                          lower=True,
                          split=' ',
                          oov_token='unrecognized-word',
                          char_level=False)  # prohibit tokenize every symbol

    tokenizer.fit_on_texts(txt_train)
    return tokenizer


def make_train_test(
        tokenizer,
        txt_train,
        txt_test=None
):
    # transform word to tokens
    seq_train = tokenizer.texts_to_sequences(txt_train)

    if txt_test:
        # transform word to tokens
        seq_test = tokenizer.texts_to_sequences(txt_test)
    else:
        seq_test = None

    return seq_train, seq_test


def print_text_stats(
    title,
    texts,
    sequences,
    class_labels=CLASS_LIST
):
    chars = 0
    words = 0

    print(f'Статистика по {title} текстам:')

    for cls in range(len(class_labels)):
        print(
            '{:<15} {:9} символов,{:8} слов'.format(class_labels[cls],
                                                    len(texts[cls]),
                                                    len(sequences[cls])))
        chars += len(texts[cls])
        words += len(sequences[cls])

    print('----')
    print('{:<15} {:9} символов,{:8} слов\n'.format('В сумме', chars, words))


def split_sequence(
        sequence,
        win_size,
        hop
):
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]


def vectorize_sequence(
        seq_list,
        win_size,
        hop
):
    class_count = len(seq_list)
    x, y = [], []

    for cls in range(class_count):
        vectors = split_sequence(seq_list[cls], win_size, hop)
        x += vectors

        y += [utils.to_categorical(cls, class_count)] * len(vectors)

    return np.array(x), np.array(y)


def compile_train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    optimizer='adam',
    epochs=50,
    batch_size=128,
    figsize=(20, 5)
):
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    plot_model(model, dpi=60, show_shapes=True)

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('model learning plot')
    ax1.plot(history.history['accuracy'],
               label='train accuracy')
    ax1.plot(history.history['val_accuracy'],
               label='test accuracy')
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('share of correct answers')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='train mistakes')
    ax2.plot(history.history['val_loss'],
               label='validate mistakes')
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mistakes')
    ax2.legend()
    plt.show()

    return model


def eval_model(
    model,
    x,
    y_true,  # text sample of text marks
    class_labels,
    cm_round=2,  # round parameter
    title='',
    figsize=(25, 25)
):

    y_pred = model.predict(x)

    # mistakes matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')

    # matrix round
    cm = np.around(cm, cm_round)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f'Neural network {title}: normalized mistakes matrix', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()
    plt.xlabel('prediction classes', fontsize=26)
    plt.ylabel('actual classes', fontsize=26)
    fig.autofmt_xdate(rotation=45)
    plt.show()

    print('-' * 100)
    print(f'neuralnetwork: {title}')

    for cls in range(len(class_labels)):
        # max confidence
        cls_pred = np.argmax(cm[cls])
        msg = 'true :-)' if cls_pred == cls else 'false :-('
        print('Class: {:<20} {:3.0f}% classify as {:<20} - {}'.format(
            class_labels[cls],
            100. * cm[cls, cls_pred],
            class_labels[cls_pred],
            msg
        ))

    print('\naverage accuracy: {:3.0f}%'.format(100. * cm.diagonal().mean()))

def compile_train_eval_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    class_labels=CLASS_LIST,
    title='',
    optimizer='adam',
    epochs=50,
    batch_size=128,
    graph_size=(20, 5),
    cm_size=(15, 15)
):

    model = compile_train_model(model,
                                x_train, y_train,
                                x_test, y_test,
                                optimizer=optimizer,
                                epochs=epochs,
                                batch_size=batch_size,
                                figsize=graph_size)


    eval_model(model, x_test, y_test,
               class_labels=class_labels,
               title=title,
               figsize=cm_size)


    return model

def make_mod(
    VOCAB_SIZE,
    WIN_SIZE,
    CLASS_COUNT
):

    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 10, input_length=WIN_SIZE))
    model.add(SpatialDropout1D(0.2))
    model.add(BatchNormalization())
    model.add(Conv1D(20, 3, activation='relu'))
    model.add(Conv1D(20, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(CLASS_COUNT, activation='softmax'))
    return model

VOCAB_SIZE = 20000
WIN_SIZE = 1000
WIN_HOP = 100

with timex():
    # trained tokenizer
    tok = make_tokenizer(VOCAB_SIZE, text_train)

    seq_train, seq_test = make_train_test(tok, text_train, text_test)

    print("Фрагмент обучающего текста:")
    print("В виде оригинального текста:              ", text_train[0][:101])
    print("Он же в виде последовательности индексов: ", seq_train[0][:20])

    x_train, y_train = vectorize_sequence(seq_train, WIN_SIZE, WIN_HOP)

    x_test, y_test = vectorize_sequence(seq_test, WIN_SIZE, WIN_HOP)

    print_text_stats('обучающим', text_train, seq_train)
    print_text_stats('тестовым', text_test, seq_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model_Conv_1 = make_mod(VOCAB_SIZE, WIN_SIZE, CLASS_COUNT)

mymodel = compile_train_eval_model(
    model_Conv_1,
    x_train, y_train,
    x_test, y_test,
    optimizer='adam',
    epochs=50,
    batch_size=200,
    class_labels=CLASS_LIST,
    title='Russian writters'
)

text_val = []

with open(f'rec_1d/refer/островский.txt', 'r', encoding="windows-1251") as f:  # text file reading
    text = f.read()
    text = text.replace('\n', ' ')
    text_val.append(text)
    print(text_val)

with timex():
    seq_val, _ = make_train_test(tok, text_val, None)
    x_val, _ = vectorize_sequence(seq_val, WIN_SIZE, WIN_HOP)

print(x_val.shape)

y_pred = mymodel.predict(x_val)

print(y_pred)

# Разберем результаты предсказания

r = np.argmax(y_pred, axis=1)
unique, counts = np.unique(r, return_counts=True)
counts = counts/y_pred.shape[0]*100
print(unique, counts)

for i in range(len(unique)):
  print('{:10s} - {:<.2f}%'.format(CLASS_LIST[unique[i]], counts[i]))
