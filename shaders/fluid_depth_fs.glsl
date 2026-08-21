#version 430 core
/*
 * Pass 1 Fragment: 深度写入
 * 
 * 将球形粒子的深度写入深度贴图。
 * 用 gl_PointCoord 将方形点精灵转为圆形，
 * 并计算正确的球形深度。
 */

layout (location = 0) out float out_depth;

in vec2 v_texcoord;
in float v_depth;
in vec3 v_normal;

uniform float u_near = 0.1;
uniform float u_far = 1000.0;

void main() {
    // 将方形点精灵转为圆形（丢弃圆外的像素）
    vec2 center = v_texcoord * 2.0 - 1.0;
    float r2 = dot(center, center);
    if (r2 > 1.0) {
        discard;
    }
    
    // 球形深度偏移（粒子表面的真实深度）
    // 粒子中心深度为 v_depth，球面顶点深度需要偏移
    float radius = 1.0;  // 点精灵的"半径"（归一化）
    float depth_offset = radius * sqrt(1.0 - r2);
    
    // 摄像机空间深度（粒子表面）
    float surface_depth = v_depth - depth_offset;
    
    // 非线性深度（存入纹理，供后续 pass 用）
    // 用线性深度，方便后续处理
    float linear_depth = (surface_depth - u_near) / (u_far - u_near);
    linear_depth = clamp(linear_depth, 0.0, 1.0);
    
    out_depth = linear_depth;
}
