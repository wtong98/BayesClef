# Music Composition with Bayesian Program Learning

## Running System

The core pipeline is contained in `main.py`. The other scripts also require this script
to have been run and properly cached the various objects. To execute:

```
python3 main.py
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
```
