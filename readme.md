# Camera Calibration

## Camera Models

### Brown

The projection function ($[x_w,y_w,z_w] \to [u,v]$):  
$$
\begin{align}
\pi(\bold{x}) &=\begin{bmatrix}f_x m_x \\ f_y m_y\end{bmatrix}
+\begin{bmatrix}c_x \\ c_y\end{bmatrix}\\
x&=x_w/z_w \\ y&=y_w/z_w \\
r&= \sqrt{x^2+y^2} \\
s&= 1+k_1r^2+k_2r^4+k_3r^6 \\
m_x&= sx+2xyp_1+(r^2+2x^2)p_2 \\
m_y&= sy+2xyp_2+(r^2+2y^2)p_1
\end{align}
$$  

The unproject function:  
$$
\begin{align}
\pi^{-1}(\bold{u}) &=\begin{bmatrix}x\\y\\1\end{bmatrix}\\
m_x&=\dfrac{u-c_x}{f_x}\\
m_y&=\dfrac{v-c_y}{f_y}\\
\begin{bmatrix}x\\y\end{bmatrix}&=
d^{-1}\left(\begin{bmatrix}m_x\\m_y\end{bmatrix}\right)
\end{align}
$$  
where $d^{-1}$ is the inverse function of $[m_x\ m_y]^T = d\left([x\ y]^T\right)$ in projection function.
