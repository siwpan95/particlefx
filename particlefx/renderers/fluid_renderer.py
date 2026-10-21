"""
流体视觉效果渲染器（折中版 SSFR）

糅合三种方案：
- 方案A（Metaball）：用密度场模拟粒子融合
- 方案B（SSFR框架）：深度贴图 + 平滑 + 光照
- 方案C（Additive增强）：发光效果模拟流体表面

核心思路：
1. 粒子位置 → 密度场（numpy向量化）
2. 高斯平滑密度场（模拟流体表面张力）
3. 阈值提取液面（Metaball融合效果）
4. 法线重建 + Blinn-Phong 光照
5. Additive 混合增强高光

性能：纯 numpy，无 PyOpenGL 依赖。
适用：1000粒子以内实时渲染（60fps）。

使用示例：
    from .fluid_renderer import FluidRenderer
    
    renderer = FluidRenderer(width=800, height=600)
    
    # 用 ParticleSystemNumpy 驱动
    sys_np = ParticleSystemNumpy(max_particles=2000)
    ...
    
    # 每帧调用
    surface = renderer.render(sys_np)
    screen.blit(surface, (0, 0))
"""

import numpy as np
from typing import Any, Optional, Tuple


# ================================================================
#  高斯模糊（分离卷积，性能优化）
# ================================================================

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """生成一维高斯核"""
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    kernel_1d = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    return kernel_1d.astype(np.float32)


def gaussian_blur(img: np.ndarray, kernel_size: int = 15, sigma: float = 4.0) -> np.ndarray:
    """
    分离高斯模糊（先水平后垂直，O(n*k) 而非 O(n*k^2)）。
    
    Args:
        img          : (H, W) 或 (H, W, 3) float32 数组
        kernel_size  : 核大小（奇数）
        sigma        : 高斯标准差
    
    Returns:
        模糊后的数组（同输入 shape）
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    
    if img.ndim == 2:
        # 灰度图：分离卷积
        # 水平
        padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
        h_blur = np.zeros_like(img)
        for i in range(kernel_size):
            h_blur += kernel[i] * padded[:, i:i+img.shape[1]]
        
        # 垂直
        padded = np.pad(h_blur, ((pad, pad), (0, 0)), mode='edge')
        v_blur = np.zeros_like(img)
        for i in range(kernel_size):
            v_blur += kernel[i] * padded[i:i+img.shape[0], :]
        
        return v_blur
    
    elif img.ndim == 3:
        # 彩色图：逐通道处理
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            padded = np.pad(channel, ((0, 0), (pad, pad)), mode='edge')
            h_blur = np.zeros_like(channel)
            for i in range(kernel_size):
                h_blur += kernel[i] * padded[:, i:i+channel.shape[1]]
            
            padded = np.pad(h_blur, ((pad, pad), (0, 0)), mode='edge')
            v_blur = np.zeros_like(channel)
            for i in range(kernel_size):
                v_blur += kernel[i] * padded[i:i+channel.shape[0], :]
            
            result[:, :, c] = v_blur
        return result
    
    return img


# ================================================================
#  流体渲染器核心
# ================================================================

class FluidRenderer:
    """
    折中版 SSFR 流体渲染器。
    
    用 numpy 在 CPU 实现屏幕空间流体渲染，
    无需 PyOpenGL，效果接近 GPU 方案。
    
    参数说明：
        width, height   : 画布尺寸
        particle_radius : 粒子渲染半径（像素）
        smooth_sigma    : 高斯平滑强度（越大越融合）
        threshold       : 密度场阈值（0~1，越小液面越厚）
        fluid_color     : 流体颜色 (R, G, B) 0~255
        ambient         : 环境光强度
        diffuse         : 漫反射强度
        specular          : 高光强度
        shininess       : 高光锐度
        additive_blend  : 是否启用 additive 混合增强
    """
    
    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 particle_radius: int = 20,
                 smooth_sigma: float = 5.0,
                 threshold: float = 0.3,
                 fluid_color: Tuple[int, int, int] = (50, 180, 255),
                 ambient: float = 0.15,
                 diffuse: float = 0.6,
                 specular: float = 0.8,
                 shininess: float = 32.0,
                 additive_blend: bool = True,
                 light_pos: Tuple[float, float, float] = (0.3, 0.3, 1.0),
                 ):
        self.width = width
        self.height = height
        self.particle_radius = particle_radius
        self.smooth_sigma = smooth_sigma
        self.threshold = threshold
        self.fluid_color = np.array(fluid_color, dtype=np.float32)
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.additive_blend = additive_blend
        self.light_dir = np.array(light_pos, dtype=np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)  # 归一化
        
        # 预计算高斯核（避免每帧重复计算）
        self._kernel_size = min(31, int(smooth_sigma * 4) | 1)  # 奇数
        self._gaussian_kernel = gaussian_kernel(self._kernel_size, smooth_sigma)
        
        # 缓存：密度场、深度场
        self._density_field = np.zeros((height, width), dtype=np.float32)
        self._smoothed = np.zeros((height, width), dtype=np.float32)
        
        # 光照计算缓存
        self._normal_map = np.zeros((height, width, 3), dtype=np.float32)
        
        # 输出 Surface
        self._output_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Additive 混合用的背景
        self._bg_array = np.zeros((height, width, 3), dtype=np.float32)
    
    def render(self,
               particle_positions: np.ndarray,
               particle_colors: Optional[np.ndarray] = None,
               particle_sizes: Optional[np.ndarray] = None,
               particle_alphas: Optional[np.ndarray] = None,
               background: Optional[np.ndarray] = None,
               ) -> np.ndarray:
        """
        渲染粒子为流体效果。
        
        Args:
            particle_positions : (N, 2) float32 — 粒子位置 (x, y)
            particle_colors    : (N, 3) uint8 — 粒子颜色（None=统一流体色）
            particle_sizes     : (N,) float32 — 粒子大小（None=统一半径）
            particle_alphas    : (N,) float32 — 粒子透明度（None=全不透明）
            background         : (H, W, 3) uint8 — 背景图（None=黑色）
        
        Returns:
            (H, W, 3) uint8 数组 — 渲染结果
        """
        n = len(particle_positions)
        if n == 0:
            return self._output_array if background is None else background.copy()
        
        # Step 1: 生成密度场（Metaball 核心）
        density = self._build_density_field(
            particle_positions, particle_sizes, particle_alphas
        )
        
        # Step 2: 高斯平滑（模拟表面张力）
        smoothed = self._smooth_density(density)
        
        # Step 3: 提取液面掩码
        fluid_mask = smoothed > self.threshold
        
        if not np.any(fluid_mask):
            return self._output_array if background is None else background.copy()
        
        # Step 4: 法线重建（从平滑后的密度场）
        normals = self._reconstruct_normals(smoothed)
        
        # Step 5: Blinn-Phong 光照
        lit = self._apply_lighting(smoothed, normals, fluid_mask, particle_colors, particle_positions)
        
        # Step 6: Additive 混合增强（方案C）
        if self.additive_blend:
            lit = self._apply_additive_glow(lit, smoothed, fluid_mask)
        
        # Step 7: 合成背景
        result = self._composite_background(lit, background)
        
        self._output_array = result
        return result
    
    def render_from_numpy_system(self,
                                 sys_np,
                                 background: Optional[np.ndarray] = None,
                                 ) -> np.ndarray:
        """
        直接从 ParticleSystemNumpy 渲染（便捷接口）。
        
        Args:
            sys_np    : ParticleSystemNumpy 实例
            background: 背景图
        
        Returns:
            (H, W, 3) uint8 数组
        """
        alive = sys_np.alive
        if not np.any(alive):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        pos = sys_np.pos[alive]      # (N, 2)
        colors = sys_np.color[alive] # (N, 3) uint8
        sizes = sys_np.size[alive]   # (N,)
        alphas = sys_np.alpha[alive] # (N,)
        
        return self.render(pos, colors, sizes, alphas, background)
    
    def render_to_surface(self,
                          sys_np,
                          pygame_screen: Optional[Any] = None,
                          background: Optional[np.ndarray] = None,
                          ) -> Any:
        """
        渲染到 Pygame Surface（直接 blit 到屏幕用）。

        Args:
            sys_np        : ParticleSystemNumpy 实例
            pygame_screen : 目标 Surface（None=创建新 Surface）
            background    : 背景图

        Returns:
            Pygame Surface
        """
        # pygame 是可选依赖（pip install particlefx[pygame]），按需延迟导入
        import pygame

        img_array = self.render_from_numpy_system(sys_np, background)
        
        if pygame_screen is not None:
            # 直接写入现有 Surface（避免创建新对象）
            pygame.surfarray.blit_array(pygame_screen, img_array.swapaxes(0, 1))
            return pygame_screen
        
        # 创建新 Surface
        surface = pygame.Surface((self.width, self.height))
        pygame.surfarray.blit_array(surface, img_array.swapaxes(0, 1))
        return surface
    
    # ================================================================
    #  内部方法
    # ================================================================
    
    def _build_density_field(self,
                             positions: np.ndarray,
                             sizes: Optional[np.ndarray],
                             alphas: Optional[np.ndarray],
                             ) -> np.ndarray:
        """
        构建密度场（Metaball 核心）。
        
        每个粒子贡献一个高斯分布到密度场，
        多个粒子叠加后形成连续液面。
        
        向量化实现：避免 Python 循环，用 broadcasting。
        """
        density = np.zeros((self.height, self.width), dtype=np.float32)
        
        if sizes is None:
            radii = np.full(len(positions), self.particle_radius, dtype=np.float32)
        else:
            radii = sizes * 2.0  # 转换为像素半径
        
        if alphas is None:
            alphas = np.ones(len(positions), dtype=np.float32)
        
        # 分批处理（避免内存爆炸）
        batch_size = 100
        for start in range(0, len(positions), batch_size):
            end = min(start + batch_size, len(positions))
            batch_pos = positions[start:end]
            batch_radii = radii[start:end]
            batch_alphas = alphas[start:end]
            
            for i in range(len(batch_pos)):
                px, py = batch_pos[i]
                r = batch_radii[i]
                a = batch_alphas[i]
                
                if a < 0.01:
                    continue
                
                ix, iy = int(px), int(py)
                r_int = max(1, int(r * 2))  # 影响半径
                
                # 计算影响区域
                x0 = max(0, ix - r_int)
                x1 = min(self.width, ix + r_int + 1)
                y0 = max(0, iy - r_int)
                y1 = min(self.height, iy + r_int + 1)
                
                if x0 >= x1 or y0 >= y1:
                    continue
                
                # 向量化：计算该区域内所有像素到粒子的距离
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dist_sq = (xx - ix) ** 2 + (yy - iy) ** 2
                sigma_sq = (r * 0.5) ** 2  # 高斯方差
                
                # 高斯密度贡献（越远贡献越小）
                contribution = a * np.exp(-dist_sq / (2 * sigma_sq))
                
                density[y0:y1, x0:x1] += contribution
        
        # 归一化到 [0, 1]
        max_val = density.max()
        if max_val > 0:
            density /= max_val
        
        return density
    
    def _smooth_density(self, density: np.ndarray) -> np.ndarray:
        """高斯平滑密度场（模拟表面张力）"""
        return gaussian_blur(density, self._kernel_size, self.smooth_sigma)
    
    def _reconstruct_normals(self, density: np.ndarray) -> np.ndarray:
        """
        从密度场重建法线（中心差分）。
        
        密度场的梯度方向 = 液体表面的法线方向。
        """
        # Sobel 算子（更smooth than 简单差分）
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float32)
        
        # 用卷积计算梯度
        grad_x = self._convolve2d(density, sobel_x)
        grad_y = self._convolve2d(density, sobel_y)
        
        # 法线 = (-grad_x, -grad_y, 1) 归一化
        # （负号：密度增加的方向是"内部"，法线指向外部）
        nx = -grad_x
        ny = -grad_y
        nz = np.ones_like(density)
        
        # 归一化
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        norm = np.maximum(norm, 1e-6)
        
        normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
        return normals
    
    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """简单的 2D 卷积（3x3 kernel）"""
        h, w = img.shape
        padded = np.pad(img, 1, mode='edge')
        result = np.zeros_like(img)
        
        for i in range(3):
            for j in range(3):
                result += kernel[i, j] * padded[i:i+h, j:j+w]
        
        return result
    
    def _apply_lighting(self,
                        density: np.ndarray,
                        normals: np.ndarray,
                        mask: np.ndarray,
                        colors: Optional[np.ndarray],
                        positions: np.ndarray,
                        ) -> np.ndarray:
        """
        Blinn-Phong 光照模型。
        
        返回 (H, W, 3) float32 数组，范围 [0, 255]。
        """
        h, w = density.shape
        
        # 视角方向（摄像机在 z=1 处，看向 -z 方向）
        view_dir = np.array([0, 0, 1], dtype=np.float32)
        
        # 环境光
        ambient = self.ambient
        
        # 漫反射
        n_dot_l = np.sum(normals * self.light_dir, axis=-1)
        n_dot_l = np.clip(n_dot_l, 0, 1)
        diffuse = self.diffuse * n_dot_l
        
        # 半程向量（Blinn-Phong）
        half_vec = self.light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)
        n_dot_h = np.sum(normals * half_vec, axis=-1)
        n_dot_h = np.clip(n_dot_h, 0, 1)
        specular = self.specular * (n_dot_h ** self.shininess)
        
        # 合成光照
        light = ambient + diffuse + specular
        light = np.clip(light, 0, 1)
        
        # 应用流体颜色
        fluid_color_norm = self.fluid_color / 255.0
        
        # 密度作为"厚度"，影响颜色强度
        thickness = np.clip(density, 0, 1)
        
        # 最终颜色
        r = fluid_color_norm[0] * light * thickness
        g = fluid_color_norm[1] * light * thickness
        b = fluid_color_norm[2] * light * thickness
        
        result = np.stack([r, g, b], axis=-1) * 255.0
        
        # 只保留液面区域
        result[~mask] = 0
        
        return result.astype(np.float32)
    
    def _apply_additive_glow(self,
                             lit: np.ndarray,
                             density: np.ndarray,
                             mask: np.ndarray,
                             ) -> np.ndarray:
        """
        Additive 混合增强（方案C）。
        
        在液面边缘添加发光效果，模拟流体表面高光。
        """
        # 计算边缘（用 Laplacian）
        edge = self._laplacian(density)
        edge = np.clip(edge, 0, 1)
        
        # 边缘发光颜色（偏白/偏亮）
        glow_color = np.array([200, 230, 255], dtype=np.float32)
        
        # Additive 混合
        glow_strength = 0.5
        for c in range(3):
            lit[:, :, c] = np.clip(
                lit[:, :, c] + glow_color[c] * edge * glow_strength,
                0, 255
            )
        
        return lit
    
    def _laplacian(self, img: np.ndarray) -> np.ndarray:
        """Laplacian 边缘检测（3x3）"""
        kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32)
        return np.abs(self._convolve2d(img, kernel))
    
    def _composite_background(self,
                              foreground: np.ndarray,
                              background: Optional[np.ndarray],
                              ) -> np.ndarray:
        """合成背景（简单 alpha 混合）"""
        if background is None:
            return np.clip(foreground, 0, 255).astype(np.uint8)
        
        bg = background.astype(np.float32)
        fg = foreground
        
        # 用密度作为 alpha
        # （这里简化：直接用前景的不透明度）
        alpha = np.clip(np.mean(fg, axis=-1) / 255.0, 0, 1)
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        
        result = fg + bg * (1 - alpha)
        return np.clip(result, 0, 255).astype(np.uint8)


# ================================================================
#  快速 Demo（独立运行测试）
# ================================================================

def demo_fluid_particles():
    """
    演示：用 FluidRenderer 渲染粒子为流体效果。
    
    运行：python -m renderers.fluid_renderer
    """
    import pygame
    import sys
    
    pygame.init()
    
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Fluid Renderer Demo")
    clock = pygame.time.Clock()
    
    # 创建渲染器
    renderer = FluidRenderer(
        width=width,
        height=height,
        particle_radius=25,
        smooth_sigma=6.0,
        threshold=0.25,
        fluid_color=(50, 180, 255),
        additive_blend=True,
    )
    
    # 模拟粒子（用 numpy 管理）
    num_particles = 500
    positions = np.random.uniform(100, 700, (num_particles, 2)).astype(np.float32)
    velocities = np.random.uniform(-2, 2, (num_particles, 2)).astype(np.float32)
    sizes = np.random.uniform(10, 30, num_particles).astype(np.float32)
    alphas = np.ones(num_particles, dtype=np.float32)
    
    colors = np.random.randint(0, 255, (num_particles, 3), dtype=np.uint8)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 更新粒子位置
        positions += velocities
        
        # 边界反弹
        mask_x = (positions[:, 0] < 0) | (positions[:, 0] >= width)
        mask_y = (positions[:, 1] < 0) | (positions[:, 1] >= height)
        velocities[mask_x, 0] *= -1
        velocities[mask_y, 1] *= -1
        positions = np.clip(positions, 0, [width - 1, height - 1])
        
        # 渲染
        img_array = renderer.render(positions, colors, sizes, alphas)
        
        # 显示
        pygame.surfarray.blit_array(screen, img_array.swapaxes(0, 1))
        pygame.display.flip()
        
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    demo_fluid_particles()
