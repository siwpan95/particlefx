"""确定性随机数工具（与 light_field 的 Web 同源实现共享）。

提供：
- ``mulberry32`` / ``hash_seed``：32 位可序列化 PRNG，与 Web 端（Canvas2D）逐位对齐；
- ``derive_seed``：由系统 seed + salt（发射器序号 / 力场序号）派生一条独立确定性子流，
  供同一系统内多个发射器/力场获得互不干扰、且跨机器可复现的随机序列。

设计依据（最佳实践）：
- 现代确定性模拟（回放 / 联机 / 可复现测试）要求「固定步长 + 种子化随机」，且随机源
  不能依赖全局 ``random`` / ``Math.random``（见 DriftEngine *Determinism* 文档、
  simplified.media *Deterministic Lockstep*）。
- mulberry32 因「仅用单个 uint32 状态、可序列化、可回滚」被上述资料列为回滚安全的 PRNG。
"""

__all__ = ["mulberry32", "hash_seed", "derive_seed"]


def _imul(a: int, b: int) -> int:
    """JS ``Math.imul`` 等价：32 位整数乘法（取乘积低 32 位）。

    JavaScript 的算术/位运算隐式在 32 位上截断，Python 不会；
    必须在乘法处显式掩码，否则 rng() 会溢出成天文数字。
    """
    return (int(a) * int(b)) & 0xFFFFFFFF


def mulberry32(a: int):
    """确定性 32 位 PRNG，与 Web 端 mulberry32 逐位对齐（``>>>`` 为无符号移位）。

    返回一个无参 callable，每次调用产出 ``[0, 1)`` 区间的 float；
    内部仅维护一个 uint32 状态，便于序列化 / 回滚（``getstate`` 思路）。
    """
    a = int(a) & 0xFFFFFFFF

    def _next():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(a ^ (a >> 15), 1 | a)
        t = (t + _imul(t ^ (t >> 7), 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return _next


def hash_seed(seed: int, i: int) -> int:
    """用 ``(seed, 序号 i)`` 派生每颗粒子独立的确定性随机源。"""
    x = (int(seed) ^ (i * 0x9E3779B1)) & 0xFFFFFFFF
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    return x


def derive_seed(base: int, salt: int) -> int:
    """由系统 seed 与 salt（发射器序号 / 力场序号）派生一条独立确定性子流。

    采用与 ``mulberry32`` / ``hash_seed`` 同源的整数混合，保证同一
    ``(base, salt)`` 在任何机器上得到相同子种子，从而让「同一系统 seed + 相同发射器
    添加顺序」即可完全复现整场模拟，而各发射器之间互不干扰。
    """
    x = (int(base) ^ (int(salt) * 0x9E3779B1)) & 0xFFFFFFFF
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x = (x ^ (x >> 13)) & 0xFFFFFFFF
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    return x & 0xFFFFFFFF
