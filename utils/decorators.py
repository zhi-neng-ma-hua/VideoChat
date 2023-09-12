import time


def timeit(func):
    """
    装饰器，用于记录函数的执行时长。
    A decorator to record the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()        # 记录开始时间
        result = func(*args, **kwargs)  # 执行原始函数
        end_time = time.time()          # 记录结束时间
        print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper

