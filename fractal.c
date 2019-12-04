__constant float DIV = 4.0;
__constant float LIMIT = 256.0;

inline float Norm(float2 a)
{
  return a.x * a.x + a.y * a.y;
}
inline float2 Mult(float2 a, float2 b)
{
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);  
}
__kernel void fractal(__const float2 C, __const float2 color_map, __global float *output)
{
    int w = get_global_size(1);
    int h = get_global_size(0);
    int x = get_global_id(1);
    int y = get_global_id(0);

    float2 real = (float2)(1.5f,-1.5f);
    float2 imag = (float2)(1.0f,-1.0f);

    float2 Z;

    Z.x=(((real.x - real.y) / (float)(w)) * x + real.y);
    Z.y=(((imag.x - imag.y) / (float)(h)) * y + imag.y);

    float i = 0;

    do
    {
        Z = Mult(Z,Z) + C;
        i+=1.0f;

        if(Norm(Z) > DIV)
        {
            break;
        }

    } while (i < LIMIT);

    if (i != LIMIT)
    {
        output[(y*w)+x]=i;
    }

}