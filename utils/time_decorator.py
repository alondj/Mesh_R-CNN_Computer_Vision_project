from datetime import datetime


def time(method):
    def timed(*args, **kw):
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = te-ts
        else:
            print(f"{method.__name__} {te-ts}")
        return result
    return timed
