#version 430 core
/*
 * Pass 1: 粒子深度写入
 * 
 * 将粒子渲染为点精灵，写入深度贴图。
 * 每个粒子是一个球体（用 gl_PointCoord 计算球形深度）。
 * 
 * 输入：粒子位置 (x, y, z) — z 是深度（摄像机空间）
 * 输出：深度值到 R32F 纹理
 */

layout (location = 0) in vec3 in_pos;    // 粒子位置 (世界空间)
layout (location = 1) in float in_size;  // 粒子大小（直径）

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_point_scale;  // 点精灵大小缩放系数

out vec2 v_texcoord;
out float v_depth;
out vec3 v_normal;

void main() {
    // 转换到摄像机空间
    vec4 view_pos = u_view * vec4(in_pos, 1.0);
    vec3 eye_pos = view_pos.xyz;
    
    // 点精灵大小（屏幕空间）
    float point_size = u_point_scale * in_size / -eye_pos.z;
    gl_PointSize = max(1.0, point_size);
    
    // 投影到裁剪空间
    gl_Position = u_proj * view_pos;
    
    // 传递给 Fragment Shader
    v_texcoord = gl_PointCoord;
    v_depth = -eye_pos.z;  // 摄像机空间深度（正数）
    
    // 球形法线（用于调试，实际深度平滑后重建法线）
    vec2 center = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(center, center);
    if (r2 <= 1.0) {
        v_normal = normalize(vec3(center, sqrt(1.0 - r2)));
    } else {
        v_normal = vec3(0.0, 0.0, 1.0);
    }
}
