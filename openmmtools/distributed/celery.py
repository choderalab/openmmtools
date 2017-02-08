from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('openmmtools.distributed',
             broker='amqp://',
             backend='amqp://',
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
