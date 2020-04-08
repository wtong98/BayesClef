"""
Models for instantiating tokens from types
"""

from music21 import bar
from music21 import chord
from music21 import note
from music21 import stream

from internal.music2vec import START_WORD, END_WORD, REST_WORD

def to_token(type_seq, type_library, word2vec_engine):
    vecs = [type_library.emit(symbol) for symbol in type_seq]
    words = [word2vec_engine.decode(vec) for vec in vecs]
    token_seq = [word.split('_') for word in words]

    return token_seq

def to_score(token_seq, texture):
    print(token_seq)
    score = stream.Stream()
    for note in texture(token_seq):
        score.append(note)

    return score

def chord_texture(token_seq):
    for token in token_seq:
        if token[0] == REST_WORD:
            yield note.Rest()
        elif token[0] in (START_WORD, END_WORD):
            yield bar.Barline('double')
        else:
            yield chord.Chord(set(token))
