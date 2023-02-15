import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence
import numpy as np
from keras.utils.data_utils import pad_sequences




# Define the input and output sentences
inputs = ['I\'m John.', 'Benim adım John.']
outputs = ['Ben John.', 'My name is John.']

# Create a tokenizer for the input sentences
tokenizer_inputs = Tokenizer()
tokenizer_inputs.fit_on_texts(inputs)
input_sequences = tokenizer_inputs.texts_to_sequences(inputs)
input_sequences = pad_sequences(input_sequences, maxlen=11, padding='post')

# Create a tokenizer for the output sentences
tokenizer_outputs = Tokenizer()
tokenizer_outputs.fit_on_texts(outputs)
output_sequences = tokenizer_outputs.texts_to_sequences(outputs)
output_sequences = pad_sequences(output_sequences, maxlen=11, padding='post')

# Define the model architecture
encoder_inputs = keras.layers.Input(shape=(11,))
encoder_embedding = keras.layers.Embedding(len(tokenizer_inputs.word_index)+1, 50)(encoder_inputs)
encoder_lstm = keras.layers.LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = keras.layers.Input(shape=(11,))
decoder_embedding = keras.layers.Embedding(len(tokenizer_outputs.word_index)+1, 50)(decoder_inputs)
decoder_lstm = keras.layers.LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(len(tokenizer_outputs.word_index)+1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([input_sequences, output_sequences[:,:-1]], output_sequences[:,1:], epochs=100)


# Use the trained model to translate new sentences
def translate(sentence):
    sequence = tokenizer_inputs.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=11, padding='post')
    prediction = model.predict([sequence, sequence[:,:-1]])
    prediction = tf.argmax(prediction, axis=-1).numpy()
    output_sentence = ''
    for word in prediction[0]:
        if word == 0:
            break
        output_sentence += tokenizer_outputs.index_word[word] + ' '
    return output_sentence.strip()

# Test the translation function
print(translate("I'm John."))
print(translate("Benim adım John."))
