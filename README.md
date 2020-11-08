# Music Composition with Bayesian Program Learning
We apply the [Bayesian Program Learning](https://science.sciencemag.org/content/350/6266/1332) 
framework to a music composition context. 

For details, check out our [writeup](https://drive.google.com/file/d/1-qiL9f9q0C35HLltZ0pOTgOAutBq8K63/view?usp=sharing).

Music samples are available [here](https://drive.google.com/drive/folders/1YnEc5bMvp6mEuBeyaldk5hdHaiaz0zAG?usp=sharing).

## Running System

The core pipeline is contained in `main.py`. The other scripts also require this script
to have been run and properly cached the various objects. To execute:

```
python3 main.py
```

Or to choose between `HMM` and `GRU` based type generation run either of following:

```
python3 main.py gru
python3 main.py hmm
```

To run the GRU based generator, one must first run the above, and then run:

```
python3 gru_generator.py
```

## Computing Metric

After having run `main.py`, one can now compare a generated score to
the scores cached. One does so as follows:

```
python3 score.py path/to/generated/score.json
python3 score.py path/to/directory/of/scores/
```

## Other utilities

`python3 baseline.py` uses the currently cached training scores to do an independent
empirical drawing of words to produce a piece of music. This represents a system which
has learned nothing about how a piece of music works together.
