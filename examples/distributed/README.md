# Distributed computing example with `celery`

See [First Steps with Celery](http://docs.celeryproject.org/en/latest/getting-started/first-steps-with-celery.html).

First, set your environment variables:
```
RABBITMQ_SERVER (defaults to 'localhost')
RABBITMQ_USERNAME (defaults to 'user')
RABBITMQ_PASSWORD (defaults to 'password')
RABBITMQ_PORT (defaults to '5762')
RABBITMQ_VHOST (defaults to 'celery')
```

To run, first start one or more workers:
```bash
$ celery -A openmmtools.distributed worker -l info
```

Then run the application:
```bash
$ python distributed-example.py
```
