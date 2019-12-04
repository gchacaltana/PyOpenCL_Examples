
__kernel void minmax(__global float* vector, __global float2* partial, volatile __global  uint* index_total)
{
    
    uint num_groups = get_num_groups(0);
    uint local_size = get_local_size(0);
    uint local_index = get_local_id (0);
    uint global_index = mad24(get_group_id(0), local_size, local_index);
    
    __local float temp[1024];
    
    temp[local_index] = vector[global_index];/// cada work-item copiara un dato de memoria global a local

    barrier(CLK_LOCAL_MEM_FENCE);/// se espera a que todos los work-items terminen de copiar sus datos

    uint group_index = 0;

    if(local_index==0)/// el work-item número 0 local hará la busqueda del minmax
    {
        float2 result = (float2)(temp[0], temp[0]);

        for(uint i = 1; i < local_size;i++)/// iterar sobre memoria local es substancialmente mas rápido
        {
            result.x = max(temp[i],result.x);
            result.y = min(temp[i],result.y);
        }

        group_index = atomic_inc(index_total);/// incrementar atómicamente el indice en 1, la funcion regresa el valor del índice antes del incremento
        partial[group_index] = result;/// guardar el resultado parcial en el índice

    }

    if(group_index + 1 == num_groups)/// cada work-item verifica si ya se han encontrado todos los minmax parciales
    {

        for(uint i = 1; i < num_groups;i++)/// este work-item es el último en ejecución, se encargara de buscar el minmax final
        {
            partial[0].x = max(partial[0].x,partial[i].x);
            partial[0].y = min(partial[0].y,partial[i].y);
        }

    }
}