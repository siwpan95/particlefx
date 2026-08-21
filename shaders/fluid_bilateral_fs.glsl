#version 430 core
/*
 * Pass 2: 双边滤波（保边平滑深度贴图）
 * 
 * 双边滤波同时考虑空间距离和像素值差异，
 * 避免高斯滤波模糊掉流体边界的问题。
 * 
 * 输入：深度贴图（线性深度）
 * 输出：平滑后的深度贴图
 * 
 * 在 Fragment Shader 中实现（比 Compute Shader 兼容性更好）。
 * 如果性能不够，可以改用 Compute Shader（需要 OpenGL 4.3+）。
 */

layout (location = 0) out float out_depth;

uniform sampler2D u_depth_tex;
uniform float u_sigma_space = 2.0;   // 空间 sigma
uniform float u_sigma_depth = 0.1;   // 深度 sigma（像素值差异）
uniform int u_kernel_size = 15;       // 滤波核大小（奇数）

in vec2 v_uv;

void main() {
    ivec2 size = textureSize(u_depth_tex, 0);
    vec2 texel_size = 1.0 / vec2(size);
    
    float center_depth = texture(u_depth_tex, v_uv).r;
    
    float weighted_sum = 0.0;
    float weight_total = 0.0;
    
    int half_k = u_kernel_size / 2;
    
    for (int y = -half_k; y <= half_k; y++) {
        for (int x = -half_k; x <= half_k; x++) {
            vec2 offset = vec2(x, y) * texel_size;
            float sample_depth = texture(u_depth_tex, v_uv + offset).r;
            
            // 空间权重（高斯）
            float d_space = length(vec2(x, y));
            float w_space = exp(-0.5 * (d_space * d_space) / (u_sigma_space * u_sigma_space));
            
            // 深度权重（像素值差异）
            float d_depth = abs(sample_depth - center_depth);
            float w_depth = exp(-0.5 * (d_depth * d_depth) / (u_sigma_depth * u_sigma_depth));
            
            float weight = w_space * w_depth;
            
            weighted_sum += sample_depth * weight;
            weight_total += weight;
        }
    }
    
    if (weight_total > 0.0) {
        out_depth = weighted_sum / weight_total;
    } else {
        out_depth = center_depth;
    }
}
