# EC2211 Problem Set 2: Suggested Solutions

**Course version:** Fall 2026  
**Status:** Instructor's suggested solutions; authoritative for the course agent's intended methods, notation, numerical results, and conclusions

# Technology transfer in the Solow model

Write variables per worker as $$y_t=A_t k_t^\alpha,
    \qquad
    k_{t+1}-k_t=sA_tk_t^\alpha-\delta k_t.$$

- **Part a.** Capital per worker cannot jump because the capital stock is predetermined. Thus, $k_0$ is unchanged when technology increases. Output per worker, however, jumps immediately: $$\frac{y_0^{\mathrm{new}}}{y_0^{\mathrm{old}}}
              =\frac{1.2A_0k_0^\alpha}{A_0k_0^\alpha}=1.2.$$ Output per worker therefore increases by 20 percent on impact.

- **Part b.** In a Solow diagram with $k_t$ on the horizontal axis, the investment curve shifts from $sAk^\alpha$ to $1.2sAk^\alpha$. The depreciation line, $\delta k$, does not move. At the old steady-state capital stock $k^*$, investment now exceeds depreciation: $$1.2sA(k^*)^\alpha>\delta k^*.$$ Hence net investment is positive.

- **Part c.** Capital per worker rises gradually toward a new, higher steady state. Output per worker first jumps because of the increase in $A$ and then rises further as capital accumulates.

  The steady-state condition is $$sA(k^*)^\alpha=\delta k^*.$$ Therefore, $$k^*=\left(\frac{sA}{\delta}\right)^{1/(1-\alpha)}.$$ If $A$ rises by a factor of 1.2, $$\frac{k^{*\prime}}{k^*}=1.2^{1/(1-\alpha)} > 1.2.$$ Since $y^*=A(k^*)^\alpha$, or equivalently $y^*=(\delta/s)k^*$, the new steady-state output per worker is also higher by the factor $$\frac{y^{*\prime}}{y^*}=1.2^{1/(1-\alpha)} > 1.2.$$

- **Part d.** At the date of the technology transfer, output per worker has a one-time growth spike of 20 percent. During the subsequent transition, output per worker continues to grow because capital per worker is increasing. This transitional growth gradually declines as the economy approaches its new steady state. In the long run, both $A$ and capital per worker are constant, so the growth rate of output per worker returns to zero.

  A graph of the growth rate should therefore show a large positive observation at the date of the shock, positive but declining growth during the transition, and zero growth in the new steady state.

- **Part e.** With continuing TFP growth, a balanced growth path requires output and capital per worker to grow at the same rate. Using the standard growth-rate approximation, $$g_y=g_A+\alpha g_k
              \quad\text{and}\quad
              g_k=g_y,$$ so $$g_y=\frac{g_A}{1-\alpha}.$$

  A one-time rise in the level of $A$ eventually leaves the economy with constant $A$ and hence zero long-run growth per worker. When $A$ grows every period, the production and investment curves keep shifting upward, generating continuing capital deepening and sustained growth of output per worker.

# Research productivity in the Romer model

Recall that $$g_{A,t}
    =
    \zeta\lambda\frac{L_t}{A_t}.$$ On the initial balanced growth path, knowledge and population grow at the same rate: $$g_A=n.$$ Output per person is $$y_t=A_t^\gamma(1-\lambda),$$ so its balanced-growth-path growth rate is $$g_y=\gamma g_A=\gamma n.$$

- **Part a.** The fall in research productivity does not destroy any existing knowledge. The level of $A_t$ is therefore unchanged in 2030.

  The growth rate of knowledge, however, does fall immediately. Since $\zeta$ falls by 10 percent while $\lambda$, $L_t$, and $A_t$ are initially unchanged, $$g_{A,2030}^{\mathrm{new}}
              =
              0.9g_{A,2030}^{\mathrm{old}}
              =
              0.9n.$$ Thus knowledge continues to grow, but initially at a lower rate.

- **Part b.** Immediately after the change, knowledge grows more slowly than population. Consequently, $A_t/L_t$ begins to fall.

  This decline in $A_t/L_t$ gradually offsets the decline in research productivity. To see why, recall that $$g_{A,t}
              =
              \zeta\lambda\frac{L_t}{A_t}.$$ As $A_t/L_t$ falls, the number of researchers becomes larger relative to the existing stock of knowledge. The flow of new ideas therefore becomes larger relative to the knowledge stock, and the growth rate of knowledge gradually recovers.

  In the long run, knowledge and population must again grow at the same rate. Hence $$g_A=n.$$ The long-run growth rate of knowledge is therefore unchanged.

- **Part c.** Output per person does not jump in 2030. The stock of knowledge $A_t$ is unchanged at that date, and the fraction of workers producing goods, $1-\lambda$, is also unchanged.

  After 2030, output per person grows more slowly than it would have done without the fall in $\zeta$. Its path therefore gradually falls below the original path. As knowledge growth recovers, output growth also recovers.

  In the long run, output per person again grows at the same percentage rate as on the original path, but from a permanently lower level. A graph should therefore show:

  - no jump in output per person in 2030

  - slower growth during the transition, and

  - a permanently lower path with the same long-run percentage growth rate.

- **Part d.** Since $$y_t=A_t^\gamma(1-\lambda),$$ the growth rate of output per person is $$g_y=\gamma g_A.$$ In the long run, knowledge and population again grow at the same rate, so $$g_A=n
              \qquad\text{and}\qquad
              g_y=\gamma n.$$

  The long-run growth rate is therefore unchanged. Output per person is nevertheless permanently lower than it would have been without the decline in $\zeta$. Lower research productivity means that a given number of researchers produces fewer ideas, placing the economy on a lower path for knowledge and output.

  During the transition, knowledge grows more slowly than population. As $A_t/L_t$ falls, however, the number of researchers becomes larger relative to the existing stock of knowledge. Knowledge growth therefore gradually returns to $n$. The decline in $\zeta$ consequently has a temporary effect on growth but a permanent effect on the level of output per person.

- **Part e.** A rise in $\lambda$ moves workers from goods production to research.

  The immediate cost is that fewer workers produce goods. Since $$y_t=A_t^\gamma(1-\lambda),$$ output per person falls immediately when $1-\lambda$ falls.

  The dynamic benefit is that more researchers produce more new ideas: $$\Delta A_{t+1}=\zeta\lambda L_t.$$ Knowledge therefore grows faster during the transition and eventually reaches a higher path.

  Whether output per person is higher at a particular date depends on the balance between these two effects: fewer workers producing goods and a larger stock of knowledge. A higher research share therefore need not increase output per person immediately, or even at every later date. Its long-run growth rate nevertheless returns to $$g_y=\gamma n.$$

### Supplementary explanation for Check Your Understanding 5

A fall in population growth differs importantly from a fall in research productivity. At the time population growth changes, the current population, the number of researchers, and the existing stock of knowledge are unchanged, so knowledge growth does not fall immediately. Over time, population and the number of researchers grow more slowly. If the new population growth rate is $n'<n$, the new long-run growth rates are

$$g_A=n'\qquad\text{and}\qquad g_y=\gamma n'.$$

A fall in $\zeta$ lowers growth temporarily but leaves the long-run growth rate unchanged. A fall in $n$ permanently lowers the long-run growth rate in this model.

# From Malthus to innovation-driven growth

- **Part a.** With population initially fixed, a permanent increase in $A$ raises income per person immediately above subsistence income $y^s$. Since $$\frac{L_{t+1}}{L_t}=\frac{y_t}{y^s}>1,$$ population begins to grow. As population rises, land per person $D/L_t$ falls and income per person declines back toward $y^s$.

  The steady-state population satisfies $$y^s=A\left(\frac{D}{L^*}\right)^\lambda,$$ so $$L^*=D\left(\frac{A}{y^s}\right)^{1/\lambda}.$$ The long-run effect of higher productivity is therefore a larger population, while long-run income per person returns to subsistence.

- **Part b.** An idea is *non-rival* if one person’s or firm’s use does not reduce the amount available to others. An idea is *excludable* if its owner can prevent others from using it. Patents, secrecy, and licenses can create some excludability and allow innovators to earn a return on the cost of creating ideas. Too much excludability is socially costly because it restricts diffusion and use of a good with a very low marginal cost, and may also obstruct follow-on innovation.

- **Part c.** Creating an idea typically requires an up-front research cost, while allowing one additional user to apply an existing non-rival idea has a very low marginal cost. Under perfect competition, price is driven toward marginal cost. Revenue may then be insufficient to cover the fixed cost of research. Temporary market power allows the innovator to charge a markup or licensing fee and earn profits with which the original research cost can be recovered. The tradeoff is that prices above marginal cost restrict use of the idea.

- **Part d.** Creative destruction is the process by which new products and production methods replace older ones. An innovation raises aggregate productivity by permitting more or better output from given inputs. At the same time, it reduces demand for older technologies and can destroy the profits and market value of incumbent firms. The social gain from innovation can therefore coexist with losses for particular firms, workers, or owners of old technologies.

- **Part e.** Stronger patents can raise the private reward from innovation and may therefore encourage research. This matters because knowledge spillovers can make the private return smaller than the social return.

  But stronger protection also raises the price of using existing ideas, slows their diffusion, and can impede cumulative or follow-on innovation. Similarly, some prospect of market power may be needed to reward innovators, whereas weak competition can reduce firms’ pressure to improve, protect incumbents from entry, and allow them to obstruct technologies that threaten their rents. The relationship between patent strength, competition, innovation, and welfare is therefore not necessarily monotonic. The word “necessarily” makes the claim false.

- **Part f.** Institutions that reduce aggregate output may still create rents for politically powerful groups. Incumbent firms, elites, or political leaders may block reforms if competition or new technology would reduce their income or weaken their future political power. Potential winners may be numerous and poorly organized, whereas the losers are concentrated and able to influence policy. Moreover, promises to compensate today’s losers may not be credible. Distributional conflict and the protection of political power can therefore sustain inefficient institutions even when reform would increase total output.

# Check your understanding

1.  **False.** A one-time increase in the level of TFP creates an immediate increase and subsequent transitional growth in output per worker, but its growth rate returns to zero in the new steady state.

2.  **True.** Along the balanced growth path, capital and output per worker grow at the same rate. The growth-rate approximation therefore gives $g_y=g_A+\alpha g_y$, or $g_y=g_A/(1-\alpha)$.

3.  **False.** The fall in $\zeta$ temporarily lowers knowledge growth and permanently lowers the level of output per person, but the model’s long-run growth rate remains $g_y=\gamma n$.

4.  **True.** A larger $\lambda$ shifts labor from goods production to research: $L_y=(1-\lambda)L$ falls, while $L_a=\lambda L$ and the production of new ideas rise.

5.  **True.** Long-run growth is $g_y=\gamma n$ where $n$ is the population growth rate.

6.  **False.** Patents may strengthen incentives to create ideas, but stronger protection also restricts diffusion and follow-on use. Beyond some point, the social costs may exceed the incentive benefits.
