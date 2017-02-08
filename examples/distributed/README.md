# Distributed computing example with `celery`

See [First Steps with Celery](http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html).

To run, first start one or more workers:
```bash
$ celery -A openmmtools.distributed worker -l info
```

Then run the application:
```bash
$ python distributed-example.py
```
