#version 430 core
/*
 * Pass 3 Vertex: 全屏四边形
 * 
 * 用于后处理 pass（双边滤波、法线重建+光照）。
 * 画一个覆盖整个屏幕的四边形。
 */

out vec2 v_uv;

void main() {
    // 全屏四边形的 4 个顶点
    // 通过 gl_VertexID 生成，不需要顶点缓冲
    vec2 positions[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    
    vec2 uvs[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0)
    );
    
    int idx = gl_VertexID;
    gl_Position = vec4(positions[idx], 0.0, 1.0);
    v_uv = uvs[idx];
}
