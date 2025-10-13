# Automated Lagrangian Mechanics using PyTorch
Lagrangian mechanics provides a mathematically beautiful and conceptually elegant framework for describing the dynamics of physical systems. Instead of focusing on forces as in Newtonian mechanics, it captures motion through a energy function. This formulation simplifies the treatment of complex systems with constraints which are traditionally hard to model using Newtonian mechanics.

By simply defining the energy of the system in terms of arbitrary generalized coordinates we can automatically simulate the dynamics of the system using PyTorch. It is almost like magic.

> ~ Classical Mechanics, Taylor 2004: Some purists object that the Lagrangian approach makes life too easy, removing the need to think about the physics.

I will give a brief interlude which is rather mathematical, feel free to skip it and check the examples. You'll find the implementation on my [GitHub](http://github.com/jelle-westra/lagrangian).

## Background
Often due to symmetries or constraints it is often easier to describe a physical system's energy function in terms of generalized coordinates in comparison to a Newtonian force framework. For example a rollercoaster car, contrained to the rails. The Lagrangian $\mathcal{L}$ in terms of generalized coordinate $\mathbf{q}(t)$ is defined as:

$$
\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = T - V
$$

where $T$ is the kinetic energy and $V$ is the potential energy. [Hamilton's principle of least action](https://en.wikipedia.org/wiki/Hamilton%27s_principle) leads to the Euler-Lagrange equations:

$$
\frac{d}{dt}\nabla_{\dot{\mathbf{q}}}\mathcal{L} = \nabla_\mathbf{q}\mathcal{L}
$$

Equations of motion for the generalized coordinates can be derived using this relation. Thanks to the maturity of autograd tools like PyTorch we can automatically calculate numerically the generalized acceleration $\ddot{\mathbf{q}}.$

In the paper [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630), authors derive using the chain rule on the lhs:

$$
(\nabla_{\dot{\mathbf{q}}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})\ddot{\mathbf{q}} + 
(\nabla_{\mathbf{q}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})\dot{\mathbf{q}} =
\nabla_{\mathbf{q}}\mathcal{L}.
$$

Note, 

$$
(\nabla_{\dot{\mathbf{q}}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})_{ij} = \frac{\partial^2\mathcal{L}}{\partial\dot{q}_j\partial\dot{q}_i}
$$

and

$$
(\nabla_{{\mathbf{q}}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})_{ij} = \frac{\partial^2\mathcal{L}}{\partial{q}_j\partial\dot{q}_i}
$$

are the lower-right and the lower-left quadrant of the Hessian of $\mathcal{L}$ respectively; we have a linear system of equations which can be solved by:

$$
\ddot{\mathbf{q}} = 
(\nabla_{\dot{\mathbf{q}}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})^{-1}
[
    \nabla_\mathbf{q}\mathcal{L} - 
    (\nabla_{\mathbf{q}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})\dot{\mathbf{q}}
].
$$

Inspired by a post by [Magnus Ross](https://magnusross.github.io/posts/l1/), this can be implemented in PyTorch as:

<pre><code id="python_code">u = torch.tensor([t, *q, *qdot], requires_grad=True, dtype=torch.float64)
        
H = hessian(self.lagrangian, u)

J_L = jacobian(self.lagrangian, u)[1:self.n+1]
J_Q = jacobian(self.Q, u)[self.n+1:]

F = (J_L - J_Q) - H[self.n+1:, 1:self.n+1] @ qdot

M_inv = torch.inverse(H[self.n+1:, self.n+1:])
qddot = M_inv @ F
</code></pre>

I made the addition of the non-conservative disspative term $Q(\dot{q})$, see [Rayleigh dissipation function](https://en.wikipedia.org/wiki/Rayleigh_dissipation_function), for which we can write the Euler-Lagranage equations into:


$$
\frac{d}{dt}\nabla_{\dot{\mathbf{q}}}\mathcal{L} = \nabla_\mathbf{q}\mathcal{L} - \nabla_{\dot{\mathbf{q}}}Q.
$$

Note this has no implication on finding $\ddot{\mathbf{q}}$ other subtracting $\nabla_{\dot{\mathbf{q}}}Q$ inside the square brackets:

$$
\ddot{\mathbf{q}} = 
(\nabla_{\dot{\mathbf{q}}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})^{-1}
[
    (\nabla_\mathbf{q}\mathcal{L} - \nabla_{\dot{\mathbf{q}}}Q)- 
    (\nabla_{\mathbf{q}}\nabla_{\dot{\mathbf{q}}}^\intercal\mathcal{L})\dot{\mathbf{q}}
].
$$

This can be easily numerically integrated by defining state vector

$$
\mathbf{u} = (\mathbf{q},\dot{\mathbf{q}}) \implies 
\dot{\mathbf{u}} = (\dot{\mathbf{q}},\ddot{\mathbf{q}}).
$$

I have implemented an implicit midpoint integration scheme, which is symplectic. You will see that if you integrate using regular odeint (RK45) the system will not conserve its energy. Especially for longer simulations, the solution will drift quite rapidly.

## Minimal Example: Simple Harmonic Oscilator
Let's define the energy in terms of pendulum angle $\theta$ and using a small angle approximation such that we can compare our integration using the analytical solution:

$$
T = \frac{mL^2\dot{\theta}^2}{2};\qquad
V = mgL\theta^2
$$

<pre><code id="python_code">@dataclass
class Pendulum(LagrangianSolver):
    m: float        # [kg] mass `m` suspended
    L: float        # [m] at length `L`
    g: float=9.81   # [m/s]

    def T(self, u: torch.Tensor) -> torch.Tensor:
        (t, theta, theta_dot) = u
        return self.m * self.L**2 * theta_dot.square()/2
    
    def V(self, u: torch.Tensor) -> torch.Tensor:
        (t, theta, theta_dot) = u
        return self.m * self.g * theta.square()
</code></pre>

We let initial condition $\theta_0=0.1$ and $\dot{\theta_0}=0$, $t\in[0,300 \textrm{ s}]$ using 30k time-steps, and let $m=L=1$. Over a long period of integration the system's energy remains stable and close to the analytical solution. Yet the solution does drift a little, at $t=300\text{ s}$ the approximated phase-sift is around $\Delta t\sim 0.01\text{ s}$.

![pendulum](./assets/SHO.svg)

## Damped Simple Harmonic Oscilator

Now adding the dissipation term:

$$
Q = \frac{c\dot{\theta}^2}{2},
$$
where $c$ is the damping coefficient.


<pre><code id="python_code">@dataclass
class DampedPendulum(LagrangianSolver):
    c: float
    ...

    def Q(self, u: torch.Tensor) -> torch.Tensor:
        (t, theta, theta_dot) = u
        return self.c*theta_dot.square()/2
</code></pre>

Also agrees with the analytical solution:

![damped-SHO](./assets/damped-SHO.svg)

You can already see how powerful and general this framework is; simulating the dynamics of complicated systems will be just a matter of formulating the energy.


## Rollercoaster
## Outer Solar System

## Dzhanibekov Effect
Now to ramp up the difficulty we consider a Lagrangian system with holonomic constraints by use of Lagrange multipliers. Famously the Dzhanibekov effect is a classic example of such constrained rotational dynamics, where a rigid body in zero gravity exhibits a surprising flip around its intermediate principal axis of inertia.

<iframe width="560" height="315" src="https://www.youtube.com/embed/1x5UiwEEvpQ?si=fFfCD5EKQ7w_fluU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

We setup the system as a bit different from the T-shape above, we have two rods connected as cross (90deg, see the video below). The rods both have length $2R$ and the intersection of the rods is placed at the origin. Rod 1 has a mass $m_1$ on both ends, and rod 2 has mass $m_2$ on both ends. The energy of the system

$$
\begin{cases}
T = m_1\dot{\mathbf{r}_1}^2 + m_2\dot{\mathbf{r}_2}^2\\\
V = 0 
\end{cases}
$$
in terms of position of one of the endpoints of rod 1 $\mathbf{r}_1$ and for rod 2 $\mathbf{r}_2$, we make use of the symmetry. We are in space now, so the potential function is 0.

Since the length of the rods is constrained we have the following three constraints:

$$
\begin{cases}
\lambda_1(\mathbf{r}_1^2 - R^2) \quad&\text{radius rod 1,}\\\
\lambda_2(\mathbf{r}_2^2 - R^2) \quad&\text{radius rod 2, and}\\\
\gamma(\mathbf{r}_1\cdot\mathbf{r}_2) \quad&\text{orthogonality.}
\end{cases}
$$

I wrote a little symbolic procedure using SymPy to derive Lagrange multipliers symbolically, as long as there is not too much coupling between the coordinates it works. From this we find:

$$
\lambda_1 = -\frac{m_1}{R^2}\dot{\mathbf{r}}_1^2;\quad
\lambda_2 = -\frac{m_2}{R^2}\dot{\mathbf{r}}_2^2;\quad
\gamma = -\frac{4\dot{\mathbf{r}}_1\dot{\mathbf{r}}_2}{R^2}\frac{m_1m_2}{m_1 + m_2}.
$$

Now implementing this into our PyTorch framework:
<pre><code id="python_code">@dataclass
class Dzhanibekov(LagrangianSolver):
    R: float
    m1: float
    m2: float

    def T(self, u: torch.Tensor):
        (r1, r2, rdot1, rdot2) = u[1:].view(4, 3)

        lambda1 = -self.m1/self.R**2 * rdot1 @ rdot1
        lambda2 = -self.m2/self.R**2 * rdot2 @ rdot2
        gamma = -4*(rdot1 @ rdot2)/self.R**2 * self.m1*self.m2/(self.m1+self.m2)

        return (
            self.m1 * rdot1 @ rdot1 + self.m2 * rdot2 @ rdot2 + 
            lambda1 * (r1 @ r1 - self.R**2) +
            lambda2 * (r2 @ r2 - self.R**2) + 
            gamma   * (r1 @ r2)
        )

    def V(self, u: torch.Tensor):
        return torch.tensor(0., dtype=torch.float64)
</code></pre>

Let $m_1=1/10$ kg and $m_2=1$ kg and $R=1$ m, and initial positions $\mathbf{r}_1 = R\hat{k}$, $\mathbf{r}_2 = R\hat{i}$, and velocities $\dot{\mathbf{r}}_1 = 0$ and $\dot{\mathbf{r}}_2 = (0, 1, 1/100)$ m/s. We find the signature flip behavior:

<video 
    height="512"
    autoplay
    loop
    muted
    <source src="./assets/Dzhanibekov.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

## Chain of Coupled Pendulums