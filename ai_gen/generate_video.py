import datetime
import time


def generate_video_task():
    print('start generate_video_task')
    time.sleep(5)
    with open('test.txt', 'w') as f:
        f.write(datetime.datetime.now().strftime('%Y-%M-%d %H:%m:%s'))
        print('finished generate_video_task')
