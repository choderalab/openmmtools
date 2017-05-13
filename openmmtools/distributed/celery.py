from __future__ import absolute_import, unicode_literals
from celery import Celery
import os

server = os.getenv('RABBITMQ_SERVER', 'localhost')
username = os.getenv('RABBITMQ_USERNAME', 'user')
password = os.getenv('RABBITMQ_PASSWORD', 'password')
port = os.getenv('RABBITMQ_PORT', '5672')
vhost = os.getenv('RABBITMQ_VHOST', 'celery')

broker = 'amqp://%(username)s:%(password)s@%(server)s:%(port)s/%(vhost)s' % vars()
backend = 'amqp://%(username)s:%(password)s@%(server)s:%(port)s/%(vhost)s' % vars()

app = Celery('openmmtools.distributed',
             broker=broker,
             backend=backend,
             include=['openmmtools.distributed.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    task_serializer = 'pickle',
    result_serializer = 'pickle',
    accept_content = {'pickle'},
)

if __name__ == '__main__':
    app.start()
