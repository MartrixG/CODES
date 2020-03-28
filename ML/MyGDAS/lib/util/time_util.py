import time


def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    if return_str:
        re = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        return re
    else:
        return need_hour, need_mins, need_secs


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    re = '[{:}]'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return re
