"""
Scratch file to play around with basic concepts. Formatted as Hydrogen
notebook.

author: William Tong (wlt2115)
date: 2-23-2020
"""

# <codecell>
from music21 import corpus
from music21 import note

sampling_rate = 0.5

# <codecell>
def _to_text(score) -> str:
    notes = score.flat.getElementsByClass(note.Note)
    # TOD0: iterate notes, bin into words
    pass

def _to_word(note_stream) -> str:
    pass

# <codecell>
bach_bundle = corpus.search('bach', 'composer')
scores = [metadata.parse() for metadata in bach_bundle]

# <codecell>
