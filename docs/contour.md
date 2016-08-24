
### Derivation of the contour site energy and derivatives

Site energy:
$$\begin{align*}
  E &= \sum_s f(\epsilon_s) \psi_s^* M \psi_s \\
  &= \sum_s f(\epsilon_s) \sum_n [\psi_s]_n^* [M \psi_s]_n \\
  &= \sum_n \sum_s f(\epsilon_s) [\psi_s]_n^* [M \psi_s]_n.
\end{align*}$$

Alternatively, $\varphi_s = M^{1/2} \psi_s$, then
$$M^{-1/2} H M^{-1/2} \varphi_s = \epsilon_s \varphi_s$$
and we get
$$\begin{align}
   E &= \sum_n \sum_s f(\epsilon_s) | [\varphi_s]_n |^2 \\
   &= {\rm Tr} \bigg[ \oint f(z) \Big(z-M^{-1/2} H M^{-1/2}\Big)^{-1} dx \bigg] \\
   &= {\rm Tr} \bigg[ M^{1/2} \oint f(z) \big(zM - H\big)^{-1} dx M^{1/2} \bigg] \\
   &= {\rm Tr} \bigg[ \oint f(z) \big(zM - H\big)^{-1} dx M \bigg]
\end{align}$$
This suggests that we should take as the site energy
$$E_n = \sum_{i} w_i f(z_i) \Big( (zM-H) \backslash M[:, n] \Big)[n]$$
This is essentially the same effort to implement as when $M = Id$, but for
derivatives it gets a little bit messier.

I ignore the $\sum_i w_i f(z_i)$ and focus just on the rest. $R_z = (zM-H)^{-1}$ and
$m_n = M[:,n]$.
$$\begin{align}
   \partial_{x_m} \big[ R_z M \big]_{nn}
   &= \Big[ R_z M_{,m} - R_z (z M_{,m} - H_{,m} ) R_z M \Big]_{nn} \\
   &= e_n^* R_z M_{,m} e_n - e_n^* R_z (z M_{,m} - H_{,m}) R_z m_n \\
   &= (R_z^* e_n)^* M_{,m} e_n
            - (R_z^* e_n)^* (z M_{,m}-H_{,m}) (R_z m_n).
\end{align}$$
We already know $R_z m_n$ from the energy assembly, but now unfortunately
we also need to compute $R_z^* e_n$. But this is not a problem: because of the
 $z M - H$ is symmetric, $(zM-H)^{-1}$ is symmetric; therefore
 if ${\rm conj}$ denotes the
element-wise conjugation operator, then
 $$\big( R_z \big)^*  e_n
   = {\rm conj} \big( R_z \big)  e_n
   = {\rm conj} \big( R_z e_n \big),
 $$
 so we can solve for $R_z e_n$ using the same LU factorisation
 as for $R_z m_n$.

 Then, we only need to compute
 $$r_M := M_{,m} {\rm conj}(R_z e_n) \qquad \text{and} \qquad
    r_H := H_{,m} {\rm conj}(R_z e_n)$$
  and can then assemble
$$\partial_{x_m} \big[ R_z M \big]_{nn}
   = (r_M)^* e_n + (z^* r_M - r_H)^* (R_z m_n).$$

 **Question:** what about Krylov methods?
