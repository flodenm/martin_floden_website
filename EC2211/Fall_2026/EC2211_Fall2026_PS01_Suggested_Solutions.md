# EC2211 Problem Set 1: Suggested Solutions

**Course version:** Fall 2026  
**Status:** Instructor's suggested solutions; authoritative for the course agent's intended methods, notation, numerical results, and conclusions

# Comparing GDP across countries

- **Part a.** Converting Indian GDP at the market exchange rate gives $$\frac{195\text{ trillion rupees}}
               {70.4\text{ rupees per dollar}}
          =2.77\text{ trillion dollars}.$$ The ratio of Indian GDP to U.S. GDP is therefore $$\frac{2.77}{20.6}=0.134.$$ Thus, at the market exchange rate, Indian GDP was approximately 13.4 percent of U.S. GDP.

- **Part b.** The Indian price level was 0.303 times the U.S. price level. Expressing Indian GDP in U.S. prices therefore gives $$\frac{2.77}{0.303}=9.14\text{ trillion dollars}.$$ Relative to U.S. GDP, this is $$\frac{9.14}{20.6}
          =\frac{0.134}{0.303}
          =0.444.$$ After adjusting for differences in price levels, Indian real GDP was therefore approximately 44.4 percent of U.S. real GDP.

- **Part c.** Goods and services were cheaper in India than in the United States. A dollar converted into rupees could therefore purchase more in India than in the United States. Conversion at the market exchange rate does not capture this difference and consequently makes India’s output appear smaller. The PPP-adjusted comparison is more informative about the quantities of goods and services produced in the two countries.

# Has Finland fallen further behind?

For each variable, the annualized growth rate is calculated as $$g_X=100\left[
        \left(\frac{X_{2025}}{X_{2000}}\right)^{1/25}-1
    \right].$$ For example, annualized Finnish GDP-per-capita growth is $$g_{Y/N}^{FI}
    =100\left[
        \left(\frac{91.29}{76.90}\right)^{1/25}-1
    \right]
    =0.69\text{ percent per year}.$$

The results are reported below. Entries are percentage points per year.

|               | $g_{Y/N}$ | $g_{Y/H}$ | $g_{H/E}$ | $g_{E/N^{wa}}$ | $g_{N^{wa}/N}$ |
|:--------------|----------:|----------:|----------:|---------------:|---------------:|
| Finland       |     0.689 |     0.762 |  $-0.393$ |          0.405 |       $-0.083$ |
| Sweden        |     1.099 |     1.127 |  $-0.201$ |          0.226 |       $-0.052$ |
| United States |     1.306 |     1.546 |  $-0.064$ |       $-0.253$ |          0.080 |

1.  **Part a.** GDP per capita grew by approximately 0.69 percent per year in Finland, 1.10 percent in Sweden, and 1.31 percent in the United States. Finland’s annual growth rate was therefore $$1.306-0.689 \approx 0.6$$ percentage points lower than the U.S. growth rate.

2.  **Part b.** The main explanation for Finland’s slower growth relative to the United States is slower growth in output per hour ($Y/H$). This accounts for a gap of $$1.546-0.762 \approx 0.8$$ percentage points per year. Falling hours per worker ($H/E$) made an additional contribution to Finland’s relative weakness: hours per worker fell by 0.393 percent per year in Finland, compared with 0.064 percent in the United States. The decline in Finland’s working-age share also contributed modestly to the gap.

    The employment rate among the working-age population moved in the opposite direction and offset a large part of these differences. It rose by 0.405 percent per year in Finland but fell by 0.253 percent per year in the United States.

3.  **Part c.** Sweden’s GDP per capita grew approximately 0.41 percentage points per year faster than Finland’s. Faster growth in output per hour, contributing approximately 0.36 percentage points to the difference, is the most important component. The smaller decline in Swedish hours per worker contributes another 0.19 percentage points, and the smaller decline in the working-age share contributes approximately 0.03 percentage points.

    These effects are partly offset by the employment rate, which grew faster in Finland than in Sweden. This component offsets approximately 0.18 percentage points of Sweden’s advantage.

4.  **Part d.** Finland experienced the largest negative contribution from the working-age share: approximately $-0.083$ percentage points per year, compared with $-0.052$ in Sweden and $+0.080$ in the United States. One possible explanation is population aging. As a larger share of the population moves above working age, $N^{wa}/N$ falls. Other demographic developments, including changes in the share of children and migration, may also affect this ratio.

# What explains growth in output per hour?

Annualized growth in capital per hour is calculated using the same formula as above. For example, $$g_{K/H}^{FI}
    =100\left[
        \left(\frac{174.0}{110.4}\right)^{1/25}-1
    \right]
    =1.84\text{ percent per year}.$$

1.  **Part a.** Capital per hour grew by approximately 1.84 percent per year in Finland, 1.74 percent in Sweden, and 1.82 percent in the United States. Capital deepening was thus slightly faster in Finland than in the United States and somewhat slower in Sweden. Overall, the rates were quite similar.

2.  **Part b.** With $\alpha=1/3$, growth accounting gives $$g_A\approx g_{Y/H}-\frac{1}{3}g_{K/H}.$$ The resulting decomposition, in percentage points per year, is

    |               |                     |                   |            |
    |:--------------|--------------------:|------------------:|-----------:|
    |               | Productivity growth | Capital deepening | TFP growth |
    |               |           $g_{Y/H}$ |  $\alpha g_{K/H}$ |      $g_A$ |
    | Finland       |               0.762 |             0.612 |      0.150 |
    | Sweden        |               1.127 |             0.579 |      0.547 |
    | United States |               1.546 |             0.606 |      0.940 |

    Finland’s productivity-growth gap relative to the United States was about 0.78 percentage points per year. Differences in capital deepening do not account for this gap: the contribution from capital deepening was actually about 0.01 percentage points larger in Finland. The accounting exercise attributes approximately 0.79 percentage points to the difference in TFP growth.

    Sweden’s productivity-growth gap relative to the United States was about 0.42 percentage points per year. Approximately 0.03 percentage points are associated with slower capital deepening and approximately 0.39 percentage points with lower TFP growth.

3.  **Part c.** No. Growth accounting identifies the part of productivity growth that is not accounted for by measured capital deepening, but it does not establish why that residual differs across countries. Measured TFP may reflect technological progress, organization, institutions, the allocation of resources, omitted inputs such as human capital, measurement error, and changes in capacity utilization. The calculation is an accounting decomposition, not a causal explanation.

# Two changes in the labor force in the Solow model

In per-worker terms, the production function and law of motion are $$y_t=Ak_t^\alpha,
    \qquad
    k_{t+1}-k_t=sAk_t^\alpha-\delta k_t$$ when the labor force is constant. The initial steady state satisfies $$sA(k^*)^\alpha=\delta k^*.$$

## A one-time increase in the level of the labor force

- **Part a.** The aggregate capital stock is predetermined and therefore does not change immediately. Capital per worker falls from $k^*=K_0/L$ to $$k_0'=\frac{K_0}{L'}=k^*\frac{L}{L'}<k^*.$$ Output is not predetermined. Aggregate output rises because more labor is used with the existing capital stock: $$Y_0'=AK_0^\alpha(L')^{1-\alpha}>Y_0.$$ The increase is less than proportional to the increase in labor. Output per worker consequently falls to $$y_0'=A(k_0')^\alpha
              =y^*\left(\frac{L}{L'}\right)^\alpha<y^*.$$

- **Part b.** Neither curve in the per-worker Solow diagram shifts. The investment curve remains $sAk^\alpha$, and the depreciation line remains $\delta k$. Instead, the economy jumps to a point to the left of the unchanged steady state. At this lower capital stock per worker, investment exceeds depreciation.

- **Part c.** Positive net investment causes capital per worker to rise. Output per worker therefore also rises. Both variables gradually return to their original steady-state levels. Following the initial fall in output per worker when the labor force increases, output per worker has a positive growth rate during the transition. This growth rate gradually falls toward zero as the economy approaches the steady state.

- **Part d.** Because $A$, $s$, and $\delta$ are unchanged, steady-state capital and output per worker are unchanged. Aggregate capital and output, however, are higher in the new steady state. Since $$K^*=k^*L,\qquad Y^*=y^*L,$$ both eventually rise in proportion to the permanent increase in the labor force.

## A higher population growth rate

With labor-force growth $n$, the per-worker law of motion is $$k_{t+1}-k_t
    =\frac{sAk_t^\alpha-(\delta+n)k_t}{1+n}.$$ The steady-state condition is $$sA(k^*)^\alpha=(\delta+n)k^*.$$

- **Part e.** A rise in $n$ makes the break-even-investment line steeper: it rotates upward from $(\delta+n)k$ to $(\delta+n')k$. The investment curve $sAk^\alpha$ does not shift. The new steady-state capital stock per worker is lower.

- **Part f.** Immediately after the change, the level of capital per worker is still at the old steady state, but investment is no longer sufficient to maintain it. Capital and output per worker therefore fall during the transition. They approach new, permanently lower steady-state levels. With constant TFP, their growth rates are negative during the transition and converge to zero in the new steady state.

- **Part g.** With constant TFP, output per worker is constant in the steady state. Aggregate output $Y=yL$ therefore grows at the same rate as the labor force. Its long-run growth rate rises from $n$ to $n'$.

- **Part h.** A one-time increase in the *level* of the labor force initially dilutes the capital stock, but it does not change the per-worker steady state. Capital and output per worker eventually return to their original levels. A permanent increase in the labor force’s *growth rate* changes the amount of investment needed to maintain capital per worker. It therefore permanently lowers steady-state capital and output per worker, while raising the long-run growth rate of aggregate output.

# Check your understanding

1.  **True.** When a country has lower prices, conversion at the market exchange rate understates how many goods and services its income can purchase relative to the United States. A PPP adjustment raises its measured real GDP relative to the United States.

2.  **True.** With $Y/H=A(K/H)^\alpha$, the growth contribution from capital deepening is weighted by the output elasticity of capital, $\alpha$.

3.  **False.** TFP is calculated as a residual. In addition to technological progress, it may capture organization, omitted inputs, measurement error, resource allocation, and capacity utilization.

4.  **False.** A one-time increase in the labor force lowers output per worker initially, but it does not change the per-worker steady state when $A$, $s$, and $\delta$ remain constant.

5.  **True.** A higher population growth rate raises break-even investment and lowers steady-state output per worker. With constant TFP, aggregate output grows at the population growth rate in the steady state, so its long-run growth rate rises.
