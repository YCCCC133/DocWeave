"""Full-domain BF16 activation tabulation with the original MLX evaluation.

This is inference-only memoization of a scalar function over its COMPLETE
16-bit input domain. No interpolation, quantization, clipping or polynomial
replacement is introduced. Other dtypes retain the checkpoint's operator.
"""
import threading
import mlx.core as mx
import mlx.nn as nn

_tables = {}
_table_lock = threading.Lock()


class ExactBF16GELU(nn.GELU):
    def __init__(self, approx='none'):
        super().__init__(approx=approx)
        self._converter_approx = approx

    def __call__(self, x):
        if x.dtype != mx.bfloat16:
            return super().__call__(x)
        mode = self._converter_approx
        table = _tables.get(mode)
        if table is None:
            with _table_lock:
                table = _tables.get(mode)
                if table is None:
                    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
                    # Use precisely the same function, including its compiled
                    # rounding and special-value behavior, as the original.
                    table = nn.GELU(approx=mode)(bits.view(mx.bfloat16))
                    mx.eval(table)
                    _tables[mode] = table
        return mx.take(table, x.view(mx.uint16))
