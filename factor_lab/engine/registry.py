import inspect
from ..definitions import base
from .. import definitions

def get_all_factor_classes():
    '''反射获取definitions中定义的所有因子类'''
    factors = {}
    # 遍历 definitions 模块下的所有成员
    for name, obj in inspect.getmembers(definitions):
        if inspect.isclass(obj) and issubclass(obj, base.BaseFactor) and obj is not base.BaseFactor:
            # 实例化因子类
            instance = obj()
            factors[instance.name] = instance
    return factors