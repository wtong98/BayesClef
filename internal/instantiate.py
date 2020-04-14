"""
Models for instantiating tokens from types
"""

from music21 import bar
from music21 import chord
from music21 import note
from music21 import stream
from music21 import tie

from internal.music2vec import START_WORD, END_WORD, REST_WORD


def to_token(type_seq, type_library, word2vec_engine):
    vecs = [type_library.emit(symbol) for symbol in type_seq]
    words = [word2vec_engine.decode(vec) for vec in vecs]
    token_seq = [word.split('_') for word in words]

    return token_seq


def to_score(token_seq, texture, **texture_args):
    score = stream.Stream()
    for note in texture(token_seq, **texture_args):
        score.append(note)

    return score


def chord_texture(token_seq, duration=1):
    for token in token_seq:
        if token[0] == REST_WORD:
            yield note.Rest()
        elif token[0] in (START_WORD, END_WORD):
            yield bar.Barline('double')
        else:
            yield chord.Chord(set(token), quarterLength=duration)


def chord_with_ties_texture(token_seq, duration=1):
    chords = chord_texture(token_seq, duration)
    chords_with_ties = []

    prev = next(chords)
    for stack in chords:
        if type(stack) is chord.Chord:
            for elem in prev:
                if elem in stack:
                    elem.tie = tie.Tie()
            chords_with_ties.append(prev)
            prev = stack

    chords_with_ties.append(prev)
    return chords_with_ties

def melody_texture(token_seq, duration=0.25, use_last=False):
    for token in token_seq:
        if token[0] == REST_WORD:
            yield note.Rest()
        elif token[0] in (START_WORD, END_WORD):
            yield bar.Barline('double')
        else:
            if use_last:
                yield(note.Note(token[-1], quarterLength=duration))
            else:
                for elem in token:
                    yield(note.Note(elem, quarterLength=duration))


def piano_texture(token_seq, duration=1):
    rh = stream.Voice(melody_texture(token_seq, duration/4))
    lh = stream.Voice(chord_texture(token_seq, duration))
    for note in stream.Score([rh, lh]):
        yield note
