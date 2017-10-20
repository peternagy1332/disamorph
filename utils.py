import sys


def update_progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\t[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()