"""
对象池（ParticlePool）与空间哈希（SpatialHashGrid）单元测试（v5.0 增强 ①）
"""
import math
import random
import pytest

from particlefx.core.particle import (
    Particle, ParticlePool, ParticleSystem, Emitter, ForceField,
    CollisionBox, CollisionBoundary,
)
from particlefx.core.spatial_hash import SpatialHashGrid


# ================================================================
#  SpatialHashGrid
# ================================================================

class TestSpatialHashGridBasics:
    def test_insert_and_query_point(self):
        grid = SpatialHashGrid(cell_size=50.0)
        box = CollisionBox(x=100, y=100, width=40, height=40)
        grid.insert_rect(box, box.x, box.y, box.width, box.height)
        assert box in grid.query_point(120, 120)
        assert box not in grid.query_point(10, 10)

    def test_rect_spans_multiple_cells(self):
        grid = SpatialHashGrid(cell_size=10.0)
        box = CollisionBox(x=5, y=5, width=25, height=25)
        grid.insert_rect(box, box.x, box.y, box.width, box.height)
        # 覆盖 3×3 格，边界附近的点都应查到
        for px, py in [(6, 6), (29, 29), (15, 15), (5, 34), (34, 5)]:
            assert box in grid.query_point(px, py)
        assert box not in grid.query_point(45, 45)

    def test_remove_rect(self):
        grid = SpatialHashGrid(cell_size=50.0)
        box = CollisionBox(x=0, y=0, width=100, height=100)
        grid.insert_rect(box, 0, 0, 100, 100)
        grid.remove_rect(box)
        assert grid.query_point(50, 50) == []

    def test_query_rect_dedupes(self):
        grid = SpatialHashGrid(cell_size=50.0)
        box = CollisionBox(x=0, y=0, width=200, height=50)
        grid.insert_rect(box, 0, 0, 200, 50)
        found = grid.query_rect(0, 0, 200, 50)
        assert found.count(box) == 1
        assert len(found) == 1

    def test_clear(self):
        grid = SpatialHashGrid(cell_size=50.0)
        box = CollisionBox(x=0, y=0, width=100, height=100)
        grid.insert_rect(box, 0, 0, 100, 100)
        grid.clear()
        assert grid.query_point(50, 50) == []

    def test_point_insert_remove(self):
        grid = SpatialHashGrid(cell_size=50.0)
        p = Particle(x=10, y=10, vx=0, vy=0, life=1, max_life=1, size=1, color=(255,) * 3)
        grid.insert_point(p, p.x, p.y)
        assert p in grid.query_point(10, 10)
        grid.remove_point(p)
        assert grid.query_point(10, 10) == []

    def test_iter_nearby_pairs(self):
        grid = SpatialHashGrid(cell_size=20.0)
        ps = [Particle(x=5 + i * 6, y=5, vx=0, vy=0, life=1, max_life=1, size=1, color=(255,) * 3)
              for i in range(4)]
        for p in ps:
            grid.insert_point(p, p.x, p.y)
        pairs = list(grid.iter_nearby_pairs(radius=8.0))
        # 相邻对 (5,11),(11,17),(17,23) 距离 6 ≤ 8；隔一对距离 12 > 8
        pair_set = {frozenset((id(a), id(b))) for a, b, _ in pairs}
        assert frozenset((id(ps[0]), id(ps[1]))) in pair_set
        assert frozenset((id(ps[0]), id(ps[2]))) not in pair_set
        # 距离平方校验
        for a, b, d2 in pairs:
            dx = a.x - b.x
            dy = a.y - b.y
            assert abs(d2 - (dx * dx + dy * dy)) < 1e-9


# ================================================================
#  ParticlePool
# ================================================================

class TestParticlePool:
    def test_acquire_release_cycle(self):
        pool = ParticlePool(capacity=10)
        p1 = pool.acquire(x=1, y=2, vx=3, vy=4, life=1.0, max_life=2.0,
                          size=3.0, color=(10, 20, 30))
        assert p1 is not None
        assert (p1.x, p1.y, p1.vx, p1.vy) == (1, 2, 3, 4)
        assert pool.active_count == 1
        assert pool.available == 9
        pool.release(p1)
        assert pool.active_count == 0
        assert pool.available == 10

    def test_identity_reuse(self):
        pool = ParticlePool(capacity=5)
        p1 = pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1,
                          size=1, color=(1, 1, 1))
        pool.release(p1)
        p2 = pool.acquire(x=9, y=9, vx=9, vy=9, life=1, max_life=1,
                          size=1, color=(2, 2, 2))
        assert p2 is p1  # 复用同一实例
        assert (p2.x, p2.y) == (9, 9)

    def test_reset_clears_optional_fields(self):
        pool = ParticlePool(capacity=5)
        p = pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1, size=1,
                         color=(1, 1, 1),
                         rotation=1.5, angular_velocity=2.5,
                         trail_enabled=True, trail_max_length=8,
                         emissive=(9, 9, 9), emissive_strength=3.0,
                         _sub_emit_on_death='X')
        p.trail_history.append((1.0, 1.0))
        pool.release(p)
        q = pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1, size=1,
                         color=(2, 2, 2))
        assert q.rotation == 0.0
        assert q.angular_velocity == 0.0
        assert q.trail_enabled is False
        assert q.trail_history == []
        assert q.emissive_strength == 0.0
        assert q._sub_emit_on_death is None

    def test_exhaustion_returns_none(self):
        pool = ParticlePool(capacity=2)
        a = pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1, size=1, color=(1,) * 3)
        b = pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1, size=1, color=(1,) * 3)
        assert a is not None and b is not None
        assert pool.acquire(x=0, y=0, vx=0, vy=0, life=1, max_life=1, size=1, color=(1,) * 3) is None

    def test_reset_requires_new_values(self):
        pool = ParticlePool(capacity=2)
        with pytest.raises(TypeError):
            pool.acquire()  # 缺少必要字段


# ================================================================
#  ParticleSystem 集成
# ================================================================

class TestParticleSystemPool:
    def _make_sys(self, **kw):
        return ParticleSystem(max_particles=200, **kw)

    def test_pool_emission_reuses_instances(self):
        ps = self._make_sys(use_pool=True)
        e = Emitter(x=100, y=100, rate=1000, life=2.0)
        ps.add_emitter(e)
        assert e.particle_factory == ps._pool.acquire
        ps.update(0.1)
        assert len(ps.particles) > 0
        assert ps._pool.active_count == len(ps.particles)

    def test_pool_death_returns_instance(self):
        ps = self._make_sys(use_pool=True)
        e = Emitter(x=100, y=100, rate=1000, life=0.2)
        ps.add_emitter(e)
        ps.update(0.1)          # 发射一批
        assert ps.get_particle_count() > 0
        assert ps._pool.active_count == ps.get_particle_count()
        ps.emitters.clear()     # 停止发射
        for _ in range(20):
            ps.update(0.1)      # 全部死亡后归还
        assert ps.get_particle_count() == 0
        assert ps._pool.available == ps._pool.capacity
        assert ps._pool.active_count == 0

    def test_pool_and_nonpool_behave_identically(self):
        """对象池开/关应产生相同的运动学结果（同种子）"""
        def run(use_pool):
            random.seed(42)
            ps = ParticleSystem(max_particles=500, use_pool=use_pool)
            e = Emitter(x=200, y=200, rate=200, spread=1.0,
                        life=1.0, size=2.0, speed=30.0)
            ps.add_emitter(e)
            ps.add_force_field(ForceField(field_type='gravity', strength=50.0))
            positions = []
            for _ in range(30):
                ps.update(0.016)
                positions.append(sorted((round(p.x, 4), round(p.y, 4)) for p in ps.particles))
            return positions
        assert run(False) == run(True)

    def test_clear_releases_pool(self):
        ps = self._make_sys(use_pool=True)
        e = Emitter(x=100, y=100, rate=1000, life=1.0)
        ps.add_emitter(e)
        ps.update(0.1)
        assert ps._pool.active_count > 0
        ps.clear()
        assert ps._pool.active_count == 0
        assert ps.get_particle_count() == 0


class TestParticleSystemSpatialHash:
    def test_grid_collision_parity_with_linear(self):
        """空间哈希开/关：碰撞结果必须完全一致"""
        def run(use_grid):
            random.seed(7)
            ps = ParticleSystem(max_particles=300, use_spatial_hash=use_grid,
                                spatial_hash_cell_size=40.0)
            boxes = [
                CollisionBox(x=150, y=150, width=60, height=60, restitution=0.8),
                CollisionBox(x=400, y=300, width=80, height=40, restitution=0.5),
            ]
            for b in boxes:
                ps.add_collision_box(b)
            e = Emitter(x=300, y=300, rate=500, spread=math.pi * 2,
                        life=0.8, speed=120.0)
            ps.add_emitter(e)
            snapshots = []
            for _ in range(40):
                ps.update(0.016)
                snapshots.append([(round(p.x, 3), round(p.y, 3),
                                   round(p.vx, 3), round(p.vy, 3))
                                  for p in ps.particles])
            return snapshots
        assert run(False) == run(True)

    def test_query_neighbors_matches_linear(self):
        random.seed(11)
        ps = ParticleSystem(max_particles=500, use_spatial_hash=True,
                            spatial_hash_cell_size=25.0)
        e = Emitter(x=250, y=250, rate=300, spread=math.pi * 2, life=2.0, speed=60.0)
        ps.add_emitter(e)
        for _ in range(5):
            ps.update(0.05)
        # 空间哈希结果与手动线性扫描一致
        for cx, cy in [(250, 250), (200, 240), (310, 290)]:
            r = 60.0
            got = {id(p) for p in ps.query_neighbors(cx, cy, r)}
            want = {id(p) for p in ps.particles
                    if (p.x - cx) ** 2 + (p.y - cy) ** 2 <= r * r}
            assert got == want

    def test_query_neighbors_without_grid_falls_back(self):
        ps = ParticleSystem(max_particles=100)
        ps.particles.append(Particle(x=10, y=10, vx=0, vy=0, life=1, max_life=1,
                                     size=1, color=(1,) * 3))
        assert len(ps.query_neighbors(10, 10, 5)) == 1
        assert ps.query_neighbors(100, 100, 5) == []

    def test_no_overhead_when_disabled(self):
        ps = ParticleSystem(max_particles=100)
        assert ps._pool is None
        assert ps._grid is None
        e = Emitter(x=0, y=0)
        ps.add_emitter(e)
        assert e.particle_factory is None  # 未启用池时不注入工厂
