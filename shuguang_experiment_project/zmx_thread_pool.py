# -*- coding:utf-8 -*-

import Queue
import threading
from time_wrapper import *


"""
    经测试,字典可以异步访问!
"""
@time_recorder
def look_up_dict(d, i):
    time.sleep(20)
    print "this is thread {0}".format(i)
    for k, v in d.iteritems():
        print "key:{0}, value:{1}".format(k, v)


class WorkManager(object):
    def __init__(self):
        self.work_queue = Queue.Queue()
        self.threads = []
        # self.__init_work_queue(work_num)
        # self.__init_thread_pool(thread_num)

    """
        初始化线程池
    """
    def init_thread_pool(self, thread_num):
        for i in range(thread_num):
            self.threads.append(Worker(self.work_queue))

    """
        初始化工作队列,在实际应用中不需要调用这个方法,直接调用add_job向任务队列里添加函数
        add_job之后调用init_thread_pool,线程就可以自动从任务队列中取任务,直到任务队列中的任务全部完成.
    """
    def __init_work_queue(self, jobs_num):
        for i in range(jobs_num):
            self.add_job(do_job, i)

    """
        添加一项工作入队
    """
    def add_job(self, func, *args, **kwargs):
        # 任务入队，Queue内部实现了同步机制
        self.work_queue.put((func, args, kwargs))

    """
        检查剩余队列任务
    """
    def check_queue(self):
        return self.work_queue.qsize()

    """
        等待所有线程运行完毕
    """
    def wait_allcomplete(self):
        for item in self.threads:
            if item.isAlive():
                item.join()


class Worker(threading.Thread):
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.start()

    def run(self):
        # 死循环，从而让创建的线程在一定条件下关闭退出
        while True:
            try:
                # 任务异步出队，Queue内部实现了同步机制
                do, args, kwargs = self.work_queue.get(block=False)
                do(*args, **kwargs)
                # 通知系统任务完成
                self.work_queue.task_done()
            except Exception, e:
                print str(e)
                break


if __name__ == '__main__':
    work_manager = WorkManager()
    for i in range(10):
        work_manager.add_job(look_up_dict, d, i)
    work_manager.init_thread_pool(10)
    work_manager.wait_allcomplete()

