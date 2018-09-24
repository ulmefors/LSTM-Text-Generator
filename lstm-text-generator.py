"""Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
"""

from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

from datetime import datetime
import os
import time

# Options
CORPUS = 'nietzsche'
PRINT_TEXT = True

# I/O
CORPUS_DIR = 'data'
TB_LOG_DIR = 'logdir'
OUTPUT_DIR = 'output'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, CORPUS + '.txt')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Constants
DATETIME_FORMAT = '%y-%m-%d_%H-%M'

# Parameters
MAX_LEN = 40
CHAR_STEP = 3
DIVERSITIES = [0.2, 0.4, 0.7, 1.0]  # High number: More randomness
GENERATION_LEN = 400

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 0.01


def get_path(corpus='nietzsche'):
    """ Choose corpus for training

    :param corpus: {'nietzsche', 'donquijote', 'shakespeare', 'les-miserables',
        'petit-prince', 'hemingway'}, default 'nietzsche'
    :return: path
        Path to selected corpus
    """
    if corpus == 'donquijote':
        path = os.path.join(CORPUS_DIR, 'donquijote.txt')
    elif corpus == 'hemingway':
        path = os.path.join(CORPUS_DIR, 'hemingway.txt')
    elif corpus == 'shakespeare':
        path = os.path.join(CORPUS_DIR, 'shakespeare.txt')
    elif corpus == 'les-miserables':
        path = os.path.join(CORPUS_DIR, 'les-miserables.txt')
    elif corpus == 'petit-prince':
        path = os.path.join(CORPUS_DIR, 'petit-prince.txt')
    elif corpus == 'nietzsche':
        origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
        path = get_file('nietzsche.txt', origin=origin)
    else:
        raise ValueError('Choose valid corpus')
    return path


with io.open(get_path(corpus=CORPUS), encoding='utf-8') as f:
    # Use lower case to reduce dictionary size
    text = f.read().lower()
print('Corpus length:', len(text))

# Create lookup tables for character and index
chars = sorted(list(set(text)))
print('Total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Cut the text in semi-redundant sequences of MAX_LEN characters
sentences = []
next_chars = []
for i in range(0, len(text) - MAX_LEN, CHAR_STEP):
    sentences.append(text[i: i + MAX_LEN])
    next_chars.append(text[i + MAX_LEN])
print('Number sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), MAX_LEN, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(MAX_LEN, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])


def sample(preds, diversity=1.0):
    """ Helper function to sample an index from a probability array

    :param preds: list of int
        Prediction indeces
    :param diversity: float
        Degree of creativity/randomness
    :return: int
        Sampled index
    """
    preds = np.asarray(preds).astype('float64')
    preds = preds**(1 / diversity)
    preds = preds / np.sum(preds)
    probas = np.random.multinomial(n=1, pvals=preds, size=1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    epoch = epoch + 1  # Start count at 1

    # Do not log text generation for all epochs
    log_result = epoch <= 5 or epoch % 10 == 0
    if not log_result:
        return

    print()
    print('*** Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - MAX_LEN - 1)

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as output_file:
        for diversity in DIVERSITIES:
            generated = ''
            sentence = text[start_index: start_index + MAX_LEN]
            seed = sentence[:]
            generated += sentence

            if PRINT_TEXT:
                print('\n')
                print('*** Diversity:', diversity)
                print('*** Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

            for i in range(GENERATION_LEN):
                x_pred = np.zeros((1, MAX_LEN, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                if PRINT_TEXT:
                    sys.stdout.write(next_char)
                    sys.stdout.flush()

            output_text = '\n\nEpoch %s, Diversity %s, Seed: %s' \
                          '\nGenerated: %s'\
                          % (epoch, diversity, seed, generated)
            try:
                output_file.write(output_text)
            except UnicodeEncodeError as e:
                print('UnicodeError', e)
            except Exception as e:
                print('Other Exception', e)
        if PRINT_TEXT:
            print('\n')


# Print callback
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Tensorboard callback
timestamp = datetime.fromtimestamp(time.time()).strftime(DATETIME_FORMAT)
log_dir = os.path.join(TB_LOG_DIR, timestamp)
tb_callback = TensorBoard(log_dir=log_dir)

# Train model
model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[print_callback, tb_callback])
