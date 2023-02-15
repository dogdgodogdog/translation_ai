#Beam Search:
#Beam search is a search algorithm used to generate the most likely sequence of words in a translation model. It keeps track of the top k most likely translations at each step, where k is a hyperparameter called the beam width. Here's an example of how you could implement beam search in the previous script:

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

decoder_inputs = keras.layers.Input(shape=(1,))
decoder_embedding = keras.layers.Embedding(len(tokenizer_outputs.word_index)+1, 50)(decoder_inputs)
decoder_lstm = keras.layers.LSTM(50, return_sequences=True, return_state=True)
decoder_dense = keras.layers.Dense(len(tokenizer_outputs.word_index)+1, activation='softmax')

# Define the encoder model
encoder_model = keras.models.Model(encoder_inputs, encoder_states)

# Define the decoder model
decoder_state_input_h = keras.layers.Input(shape=(50,))
decoder_state_input_c = keras.layers.Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    output_sequence = []
    
    while len(sequences) > 0:
        temp = []
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Get the last character in the sequence
            target_seq = np.array(seq).reshape(1,-1)
            target_seq = pad_sequences(target_seq, maxlen=11, padding='post')
            last_char = target_seq[0][-1]
            # If the last character is the end-of-sequence token, add the sequence to the output list
            if last_char == 0:
                output_sequence.append([seq, score])
            else:
                # Get the decoder state for the current sequence
                states_value = encoder_model.predict(data)
                # Generate the next word probabilities and the decoder state for the current sequence
                target_seq = np.array([last_char]).reshape(1, 1)
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                # Get the top k words with the highest probabilities
                word_probs = output_tokens[0, -1, :]
                top_k_indexes = np.argsort(word_probs)[-k:]
                for j in top_k_indexes:
                    # Create a new sequence with the new word and the updated score
                    new_seq = seq + [j]
                    new_score = score - np.log(word_probs[j])
                    temp.append([new_seq , new_score])
        # Sort the sequences by score
        sequences = sorted(temp, key=lambda x: x[1])
        # Keep only the top k sequences
        sequences = sequences[:k]
    return output_sequence

# Test the beam search decoder
input_sentence = np.array(input_sequences[0]).reshape(1,-1)
beam_width = 3
output = beam_search_decoder(input_sentence, beam_width)
print('Input sentence:', inputs[0])
print('Output sequences (beam width = {}):'.format(beam_width))
for seq in output:
    # Remove the start-of-sequence token and the end-of-sequence token
    output_sentence = [tokenizer_outputs.index_word[i] for i in seq[0][1:-1]]
    output_sentence = ' '.join(output_sentence)
    print(output_sentence)
    
def translate_beam_search(sentence, k):
    sequence = tokenizer_inputs.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=11, padding='post')
    encoder_output, state_h, state_c = encoder_model.predict(sequence)
    states_value = [state_h, state_c]
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tokenizer_outputs.word_index['start']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        output_tokens = beam_search_decoder(output_tokens[0], k)
        target_seq = np.zeros((1, 1))
        output_sequences = []
        for i in range(k):
            output_token = output_tokens[i][0][-1]
            output_sequences.append(output_tokens[i][0])
            target_seq[0, 0] = output_token
            if output_token == tokenizer_outputs.word_index['end'] or len(output_sequences) >= 11:
                stop_condition = True
                break
        decoded_sentence = ' '.join([tokenizer_outputs.index_word[w] for w in output_sequences[0]])
        states_value = [h, c]
    return decoded_sentence

# Test the beam search translation function
print(translate_beam_search("I'm John.", 3))
print(translate_beam_search("Benim adım John.", 3))


#In this script, the beam_search_decoder function takes in a matrix of output probabilities and returns the k most likely sequences of output words using beam search. The translate_beam_search function uses this beam search method to generate the most likely translation of a given input sentence.

#To use the beam search translation function, simply call it with an input sentence and a beam width, like this:
translate_beam_search("I'm John.", 3)
#This will return the translated sentence using beam search with a beam width of 3.
