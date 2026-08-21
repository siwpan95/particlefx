"""
均匀网格空间哈希（Spatial Hash Grid）

加速场景中的碰撞检测与邻居查询：
- 把空间划分为 cell_size×cell_size 的均匀网格
- 矩形/圆形对象登记到覆盖的格子中
- 点查询只检查所在格子内的候选对象（O(1) 平均）

设计原则：
- 纯 Python + dict，无外部依赖
- 支持点对象（单格）与矩形对象（多格）混存
- 查询结果去重；remove 操作按登记时记录反查

用法：
    from core.spatial_hash import SpatialHashGrid

    grid = SpatialHashGrid(cell_size=64.0)
    grid.insert_rect(box, box.x, box.y, box.width, box.height)
    candidates = grid.query_point(px, py)   # 点所在格子的矩形对象
    grid.remove_rect(box)
    grid.clear()
"""

from typing import Any, Dict, List, Optional, Tuple


class SpatialHashGrid:
    """均匀网格空间哈希（v5.0 新增 — 碰撞/邻居查询加速）"""

    def __init__(self, cell_size: float = 64.0):
        self.cell_size = max(1e-3, float(cell_size))
        # 格子 → 对象列表（重建时复用 dict，减少分配）
        self._cells: Dict[Tuple[int, int], List[Any]] = {}
        # 对象 → 所在格子列表（用于 remove / 多格去重）
        self._item_cells: Dict[int, List[Tuple[int, int]]] = {}
        self._count = 0

    # ------------------------------------------------------------------
    #  基础操作
    # ------------------------------------------------------------------

    def _key(self, cx: float, cy: float) -> Tuple[int, int]:
        return (int(cx // self.cell_size), int(cy // self.cell_size))

    def clear(self):
        """清空所有登记（保留容器结构，避免每帧重新分配）"""
        for cell_list in self._cells.values():
            cell_list.clear()
        self._item_cells.clear()
        self._count = 0

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    #  登记 / 注销
    # ------------------------------------------------------------------

    def insert_point(self, item: Any, x: float, y: float):
        """登记一个点对象（单格）"""
        key = self._key(x, y)
        self._cells.setdefault(key, []).append(item)
        self._item_cells[id(item)] = [key]
        self._count += 1

    def remove_point(self, item: Any):
        """注销一个点对象"""
        keys = self._item_cells.pop(id(item), None)
        if not keys:
            return
        for key in keys:
            cell_list = self._cells.get(key)
            if cell_list and item in cell_list:
                cell_list.remove(item)
        self._count = max(0, self._count - 1)

    def insert_rect(self, item: Any,
                    x: float, y: float,
                    width: float, height: float):
        """登记一个矩形对象（覆盖其所有格子）"""
        min_key = self._key(x, y)
        max_key = self._key(x + width, y + height)
        keys: List[Tuple[int, int]] = []
        for cx in range(min_key[0], max_key[0] + 1):
            for cy in range(min_key[1], max_key[1] + 1):
                key = (cx, cy)
                self._cells.setdefault(key, []).append(item)
                keys.append(key)
        self._item_cells[id(item)] = keys
        self._count += 1

    def remove_rect(self, item: Any):
        """注销一个矩形对象"""
        self.remove_point(item)  # 与点对象共用反查逻辑

    # ------------------------------------------------------------------
    #  查询
    # ------------------------------------------------------------------

    def _dedupe(self, items: List[Any]) -> List[Any]:
        """跨格查询结果去重（保持首次出现顺序）"""
        if len(items) <= 1:
            return items
        seen = set()
        result = []
        for it in items:
            key = id(it)
            if key not in seen:
                seen.add(key)
                result.append(it)
        return result

    def query_point(self, x: float, y: float) -> List[Any]:
        """查询点所在格子内的所有对象"""
        return list(self._cells.get(self._key(x, y), ()))

    def query_rect(self, x: float, y: float,
                   width: float, height: float) -> List[Any]:
        """查询与矩形重叠格子内的所有对象（去重）"""
        min_key = self._key(x, y)
        max_key = self._key(x + width, y + height)
        found: List[Any] = []
        for cx in range(min_key[0], max_key[0] + 1):
            for cy in range(min_key[1], max_key[1] + 1):
                found.extend(self._cells.get((cx, cy), ()))
        return self._dedupe(found)

    def query_circle(self, x: float, y: float,
                     radius: float) -> List[Any]:
        """查询圆形区域内的候选对象（按格子粗筛，调用方自行精确判距）"""
        return self.query_rect(x - radius, y - radius, radius * 2, radius * 2)

    def iter_nearby_pairs(self, radius: float):
        """生成相邻格子内的对象对（用于粒子间作用力，去重 + 距离过滤）

        Yields:
            (item_a, item_b, dist_sq) — 只产出 dist_sq <= radius² 的对
        """
        r2 = radius * radius
        for key, cell_items in self._cells.items():
            if len(cell_items) < 2:
                continue
            # 同一格内配对
            for i in range(len(cell_items)):
                a = cell_items[i]
                ax, ay = a.x, a.y
                for j in range(i + 1, len(cell_items)):
                    b = cell_items[j]
                    dx = ax - b.x
                    dy = ay - b.y
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        yield a, b, d2
