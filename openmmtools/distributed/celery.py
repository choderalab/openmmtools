from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('openmmtools.distributed',
             broker='amqp://test:JJEACWWXKVHZKLHZ@ec2-54-86-199-94.compute-1.amazonaws.com:5672/celery',
             backend='amqp://test:JJEACWWXKVHZKLHZ@ec2-54-86-199-94.compute-1.amazonaws.com:5672/celery',
             include=['openmmtools.distributed.tasks'])
#app = Celery('openmmtools.distributed',
#             broker='amqp://54.86.199.94',
#             backend='amqp://54.86.199.94',
#             include=['openmmtools.distributed.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    task_serializer = 'pickle',
    result_serializer = 'pickle',
    accept_content = {'pickle'},
)

if __name__ == '__main__':
    app.start()
