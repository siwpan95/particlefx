#version 430 core
/*
 * Pass 3: 法线重建 + 光照
 * 
 * 从平滑后的深度贴图重建法线（中心差分），
 * 然后做 Blinn-Phong 光照 + 折射效果。
 * 
 * 输入：
 *   - 平滑后的深度贴图
 *   - 厚度贴图（用于折射）
 * 
 * 输出：最终颜色
 */

layout (location = 0) out vec4 out_color;

uniform sampler2D u_depth_tex;
uniform sampler2D u_thickness_tex;
uniform sampler2D u_bg_tex;  // 背景纹理（用于折射）

uniform vec3 u_fluid_color = vec3(0.2, 0.7, 1.0);
uniform vec3 u_light_dir = vec3(0.3, 0.5, 1.0);
uniform vec3 u_eye_pos = vec3(0.0, 0.0, 10.0);

uniform float u_ambient = 0.15;
uniform float u_diffuse = 0.6;
uniform float u_specular = 0.8;
uniform float u_shininess = 32.0;

uniform float u_refract_strength = 0.02;  // 折射偏移强度
uniform float u_near = 0.1;
uniform float u_far = 1000.0;

in vec2 v_uv;

vec3 reconstruct_normal(vec2 uv, vec2 texel_size) {
    /*
     * 从深度贴图重建法线（中心差分）。
     * 
     * 原理：深度场的梯度方向 = 表面法线方向。
     * 用 sobel 算子计算更平滑的梯度。
     */
    
    // 采样周围深度值
    float d_l = texture(u_depth_tex, uv + vec2(-texel_size.x, 0.0)).r;
    float d_r = texture(u_depth_tex, uv + vec2( texel_size.x, 0.0)).r;
    float d_d = texture(u_depth_tex, uv + vec2(0.0, -texel_size.y)).r;
    float d_u = texture(u_depth_tex, uv + vec2(0.0,  texel_size.y)).r;
    
    // 中心差分（更简单快速）
    float dx = (d_r - d_l) / (2.0 * texel_size.x);
    float dy = (d_u - d_d) / (2.0 * texel_size.y);
    
    // 法线 = (-dx, -dy, 1) 归一化
    // （负号：深度增加的方向是"内部"）
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    
    return normal;
}

void main() {
    float depth = texture(u_depth_tex, v_uv).r;
    
    // 深度为 0 表示没有流体
    if (depth < 0.001) {
        out_color = vec4(0.0);  // 完全透明
        return;
    }
    
    ivec2 size = textureSize(u_depth_tex, 0);
    vec2 texel_size = 1.0 / vec2(size);
    
    // 重建法线
    vec3 normal = reconstruct_normal(v_uv, texel_size);
    
    // 折射：偏移背景采样坐标
    vec2 refract_offset = normal.xy * u_refract_strength;
    vec2 bg_uv = v_uv + refract_offset;
    bg_uv = clamp(bg_uv, 0.0, 1.0);
    vec3 bg_color = texture(u_bg_tex, bg_uv).rgb;
    
    // 厚度（用于 Beer-Lambert 定律）
    float thickness = texture(u_thickness_tex, v_uv).r;
    
    // Beer-Lambert：透光率 = exp(-k * d)
    vec3 attenuation = exp(-(1.0 - u_fluid_color) * thickness * 3.0);
    vec3 refracted_color = bg_color * attenuation;
    
    // Blinn-Phong 光照
    vec3 light_dir = normalize(u_light_dir);
    vec3 view_dir = normalize(u_eye_pos - vec3(v_uv, depth * u_far));
    
    // 环境光
    vec3 ambient = u_ambient * u_fluid_color;
    
    // 漫反射
    float ndotl = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = u_diffuse * ndotl * u_fluid_color;
    
    // 高光（Blinn-Phong）
    vec3 half_vec = normalize(light_dir + view_dir);
    float ndoth = max(dot(normal, half_vec), 0.0);
    float spec = pow(ndoth, u_shininess);
    vec3 spec_color = u_specular * spec * vec3(1.0);
    
    // 合成
    vec3 final_color = refracted_color * (ambient + diffuse) + spec_color;
    
    // 基于厚度设置 alpha（厚度越大越不透明）
    float alpha = clamp(thickness * 2.0, 0.0, 0.9);
    
    out_color = vec4(final_color, alpha);
}
