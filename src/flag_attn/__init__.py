try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "0.0.0"
    version_tuple = (0, 0, 0)


from flag_attn.piecewise import attention as piecewise_attn

