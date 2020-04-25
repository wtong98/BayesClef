# Music Composition with Bayesian Program Learning

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
