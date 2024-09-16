__kernel void matmul
(
    __global const float* lhs,
    __global const float* rhs,
    __global float* result,
    int lhs_cols,
    int rhs_cols
)
{
   int row = get_global_id(0);
   int col = get_global_id(1);
 
   float dot = 0;
   for (int k = 0; k < lhs_cols; ++k)
   {
      dot += lhs[row * lhs_cols + k] * rhs[k * rhs_cols + col];
   }

   result[row * rhs_cols + col] = dot;
}
