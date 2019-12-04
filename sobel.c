__constant float xOP[9]={-1,0,1,-2,0,2,-1,0,1};
__constant float yOP[9]={-1,-2,-1,0,0,0,1,2,1};

__kernel void sobel(__global const float *input, __global float *output)
{
    uint h = get_global_size(0)+2;
    uint w = get_global_size(1)+2;
    uint y = get_global_id(0)+1;
    uint x = get_global_id(1)+1;
    uint z = get_global_id(2);
        
    uint ch_size = h*w*z;
    
    float X = (input[((y-1)*w)+(x-1)+ch_size]*xOP[0]) + (input[((y-1)*w)+(x)+ch_size]*xOP[1]) + (input[((y-1)*w)+(x+1)+ch_size]*xOP[2]);
    X+= (input[((y)*w)+(x-1)+ch_size]*xOP[3]) + (input[((y)*w)+(x)+ch_size]*xOP[4]) + (input[((y)*w)+(x+1)+ch_size]*xOP[5]);
    X+= (input[((y+1)*w)+(x-1)+ch_size]*xOP[6]) + (input[((y+1)*w)+(x)+ch_size]*xOP[7]) + (input[((y+1)*w)+(x+1)+ch_size]*xOP[8]);
    X/=9;
    
    float Y = (input[((y-1)*w)+(x-1)+ch_size]*yOP[0]) + (input[((y-1)*w)+(x)+ch_size]*yOP[1]) + (input[((y-1)*w)+(x+1)+ch_size]*yOP[2]);
    Y+= (input[((y)*w)+(x-1)+ch_size]*yOP[3]) + (input[((y)*w)+(x)+ch_size]*yOP[4]) + (input[((y)*w)+(x+1)+ch_size]*yOP[5]);
    Y+= (input[((y+1)*w)+(x-1)+ch_size]*yOP[6]) + (input[((y+1)*w)+(x)+ch_size]*yOP[7]) + (input[((y+1)*w)+(x+1)+ch_size]*yOP[8]);
    Y/=9;
    
    output[(y*w)+x+ch_size] = sqrt((X*X)+(Y*Y));
}