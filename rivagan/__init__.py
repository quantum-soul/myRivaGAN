# -*- coding: utf-8 -*-

"""Top-level package for RivaGAN."""
__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev0'

from rivagan.rivagan import RivaGAN

__all__ = ('RivaGAN', )

'''
用 __all__ 暴露接口
__all__ 也是对于模块公开接口的一种约定，比起下划线，__all__ 提供了暴露接口用的”白名单“。
一些不以下划线开头的变量（比如从其他地方 import 到当前模块的成员）可以同样被排除出去。
如果显式声明了 __all__，import * 就只会导入 __all__ 列出的成员。
最后多出来的逗号在 Python 中是允许的，也是符合 PEP8 风格的。这样修改一个接口的暴露就只修改一行，方便版本控制的时候看 diff。
'''