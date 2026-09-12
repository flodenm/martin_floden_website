# Lecture Notes 5b: Consumption and Investment

**Course:** EC2211 Intermediate Macroeconomics, Stockholm University  
**Instructor:** Martin Flodén  
**Course version:** Fall 2026 (authoritative)

This is an agent-oriented rendering of `main_LN05b.tex`, checked against the 32-page `EC2211_Fall2026_LN05b.pdf`. Section and frame headings follow the lecture. Figure descriptions convey what the figures support; consult the lecture PDF for the visual itself.


### Contents and literature

- Consumption and saving

- Investment

Literature:

Jones (2024), ch. 16

Jones (2024), ch. 17.1–17.2 and the box on Tobin’s $q$ in ch. 17.3


## Consumption


### Consumption is the largest component of GDP

**Figure source:** `C_share_SE_US.png`

**Figure description:** Private final consumption expenditure as a share of GDP at current prices, Sweden and the United States, 1960–2024. The U.S. share rises from about 60% to 68%, while the Swedish share declines from about 56% to 43%. Source: OECD Economic Outlook.

Private final consumption expenditure as a percentage of GDP, current prices. Source: OECD Economic Outlook.


### Three questions about consumption

Households receive income at different points in time but can save and, sometimes, borrow.

We want to understand:

1.  How do households divide income between consumption and saving?

2.  How does consumption respond to temporary and permanent changes in income?

3.  How do interest rates and access to credit affect consumption?


### A two-period model

A household lives for two periods. Let

- $y_t$ denote disposable income in period $t$

- $c_t$ denote consumption in period $t$

- $s$ denote saving in period 1

The two budget constraints are $$\begin{aligned}
        c_1 &= y_1-s \\
        c_2 &= y_2+(1+r)s 
    
\end{aligned}$$ where $r$ is the real interest rate.

Saving can be negative: $s<0$ means that the household borrows.


### The intertemporal budget constraint

Combining the period-1 budget constraint and the period-2 budget constraint gives $$c_1+\frac{c_2}{1+r}
        =y_1+\frac{y_2}{1+r}$$

- The left-hand side is the present value of lifetime consumption

- The right-hand side is the present value of lifetime income

The household can change the timing of consumption, but not its present value.

The model assumes that the household can borrow and save at the same interest rate and ends period 2 with no debt or assets.


### Saving and the timing of consumption

**Saving:** Consume less than current income today and more than current income tomorrow $$s>0, \qquad c_1<y_1, \qquad c_2>y_2$$

**Borrowing:** Consume more than current income today and repay the loan tomorrow $$s<0, \qquad c_1>y_1, \qquad c_2<y_2$$

The interest rate determines how much future consumption must be given up to obtain one additional unit of consumption today.


### Preferences over consumption

Lifetime utility is $$U=u(c_1)+\beta u(c_2), \qquad 0<\beta\leq 1,$$ where $\beta$ measures how much the household values future utility.

We assume $$u'(c_t)>0
        \qquad\text{and}\qquad
        u''(c_t)<0$$

- More consumption raises utility.

- Marginal utility falls as consumption rises.


### Diminishing marginal utility creates a desire to smooth consumption

**Figure source:** `MACRO6_FIG16.02.jpg`

**Figure description:** Jones (2024), Figure 16.2: utility on the vertical axis and consumption on the horizontal axis. The utility curve rises but is concave. Two unequal levels of consumption, $c_1$ and $c_2$, yield lower average utility than consuming the same average amount in both periods; this illustrates consumption smoothing under diminishing marginal utility.

Figure 16.2 in Jones (2024).


### The household's choice

The household chooses $c_1$ and $c_2$ to $$\max_{c_1,c_2}\; u(c_1)+\beta u(c_2)$$ subject to $$c_1+\frac{c_2}{1+r}
        =y_1+\frac{y_2}{1+r}$$

Substituting for period-2 consumption gives a problem in one variable: $$\max_{c_1}\;
        u(c_1)+\beta u\!\left((1+r)(y_1-c_1)+y_2\right)$$


### The Euler equation

The first-order condition is $$u'(c_1)=\beta(1+r)u'(c_2).$$

At the optimum, the household is indifferent between:

- consuming one additional unit today, which gives $u'(c_1)$

- saving that unit and consuming $1+r$ additional units tomorrow, which gives $\beta(1+r)u'(c_2)$.

This condition is called the **Euler equation**.


### What determines consumption growth?

The Euler equation is $$u'(c_1)=\beta(1+r)u'(c_2).$$

Other things equal:

- A more patient household has a larger $\beta$ and chooses more consumption tomorrow relative to today

- A higher real interest rate increases the reward to saving and tends to shift consumption toward tomorrow (but see below...)

- Diminishing marginal utility prevents consumption from becoming too uneven across periods.


### The Euler equation with log utility

Suppose $$u(c_t)=\ln c_t,
        \qquad\text{so that}\qquad
        u'(c_t)=\frac{1}{c_t}$$

The Euler equation becomes $$\frac{1}{c_1}=\beta(1+r)\frac{1}{c_2}$$

Consumption is constant over time when $\beta(1+r)=1$.


### Current consumption and lifetime income

With log utility, optimal consumption in period 1 is $$c_1
        =\frac{1}{1+\beta}
        \left(y_1+\frac{y_2}{1+r}\right).$$

Suppose current income increases by $\Delta y_1$, while expected future income is unchanged. Current consumption then increases by $$\Delta c_1
        =\frac{1}{1+\beta}\Delta y_1
        <\Delta y_1.$$

The household consumes only part of the additional income today and saves the rest: $$\Delta s
        =\Delta y_1-\Delta c_1
        =\frac{\beta}{1+\beta}\Delta y_1.$$


### Temporary and permanent income changes

The result illustrates the **permanent income hypothesis**: consumption depends on expected lifetime income, not only on current income.

To make the comparison particularly simple, suppose $\beta(1+r)=1$ and income increases by $\Delta y$.

- **Temporary increase:** Only $y_1$ increases. Then $$\Delta c_1=\frac{1}{1+\beta}\Delta y<\Delta y.$$

- **Permanent increase:** Both $y_1$ and $y_2$ increase by $\Delta y$. Then $$\Delta c_1
              =\frac{1}{1+\beta}
              \left(\Delta y+\frac{\Delta y}{1+r}\right)
              =\Delta y.$$

Current consumption responds more strongly when the income increase is expected to persist.


### How does a higher interest rate affect consumption?

A higher real interest rate makes future consumption cheaper relative to current consumption.

- **Substitution effect:** Consume less today and save more.

- **Income effect:** A saver receives more interest income, while a borrower faces higher debt-service costs.

The substitution effect reduces current consumption, but the total effect also depends on whether the household is a saver or a borrower and on when it receives income.


### The benchmark model vs. the real world

The benchmark model assumes that households are forward-looking and can freely borrow or save.

In practice, households differ:

- Some cannot borrow against future income

- Some have little liquid wealth even if they own housing or retirement assets

- Some anyways respond strongly to current income or cash flow

Such households may have a high **marginal propensity to consume**: a large share of an additional unit of current income is consumed quickly.


### A borrowing constraint

Suppose the household cannot borrow, so that they must choose $c_1\leq y_1$.

If the household would prefer $c_1>y_1$:

- the borrowing constraint binds

- current consumption is tied closely to current income

- an increase in current income can generate a large increase in current consumption

The permanent-income model remains a useful benchmark, but access to credit matters for how households respond to shocks and policy.


### Swedish households have large balance sheets

**Figure source:** `assets_liabilities_SE.png`

**Figure description:** Swedish household assets, of which financial assets are shown separately, and debt, each relative to disposable income, roughly 1980–2022. Total assets rise from around 4 to over 10 times annual disposable income, while debt rises from around 0.9 to about 1.9 times. Source: Statistics Sweden.

- Household assets are large relative to disposable income

- Debt is also high, and much of it carries a variable or short fixed interest rate

- Interest-rate changes therefore redistribute cash flow across households

Household assets and liabilities as a ratio to disposable income. Source: Statistics Sweden.


### What have we learned about consumption?

- Saving and borrowing allow households to move consumption across time

- Diminishing marginal utility creates a desire to smooth consumption

- In the benchmark model, consumption depends on expected lifetime resources

- The real interest rate affects the timing of consumption

- Borrowing constraints and limited liquid wealth make current income and cash flow more important


## Investment


### Investment adds to the capital stock

Recall the capital accumulation equation from LN2: $$K_{t+1}=(1-\delta)K_t+I_t$$

Firms invest when the expected benefit from additional capital is large relative to its cost.

- The benefit depends on the marginal product of capital, $MPK$

- The cost depends on the real interest rate, depreciation, and the price of capital goods


### The return on capital

Start with a capital good whose price is constant and normalized to one.

Buying one unit of capital today produces next period:

- $MPK$ units of additional output, and

- $1-\delta$ units of remaining capital.

The total payoff is therefore $$MPK+1-\delta$$

In equilibrium, the return must equal the payoff from saving at the real interest rate: $$1+r=MPK+1-\delta$$


### The user cost of capital

Rearranging the return condition gives $$MPK=r+\delta.$$

The right-hand side is the **user cost of capital**:

- $r$ is the opportunity cost of tying up funds in capital

- $\delta$ is the loss caused by depreciation

If $MPK>r+\delta$, additional capital is profitable and the firm should invest more. As the capital stock rises, diminishing returns reduce $MPK$.


### Expected capital gains also matter

Let $p^K$ denote the price of a capital good and let $\Delta p^K/p^K$ denote its expected rate of price increase.

An approximate user-cost condition is $$MPK
        \approx
        \underbrace{r+\delta-\frac{\Delta p^K}{p^K}}
                    _{\text{user cost of capital}}$$

- A higher real interest rate raises the user cost

- Faster depreciation raises the user cost

- An expected increase in the value of capital lowers the user cost

If $MPK$ exceeds the user cost, additional investment is profitable.


### How much capital should the firm choose?

**Figure source:** `MACRO6_FIG17.01.jpg`

**Figure description:** Jones (2024), Figure 17.1: marginal product of capital $MPK$ decreases as capital $K$ increases. A horizontal user-cost line intersects $MPK$ at the desired stock of capital $K^{desired}$; lower user cost or higher expected marginal benefit increases desired capital.

Figure 17.1 in Jones (2024).


### What changes desired investment?

Firms want to invest more when:

- the real interest rate is lower

- expected future demand and productivity are higher

- investment goods are cheaper

- the tax treatment of investment is more favorable

- existing capital is low relative to the desired capital stock

Investment is forward-looking: expectations about future profits can be as important as current economic conditions.


### Tobin's $q$: a market-value perspective

Tobin’s $q$ compares $$q=\frac{\text{market value of additional installed capital}}
                 {\text{cost of purchasing and installing it}}$$

- If $q>1$, the additional capital is worth more than it costs: investment is attractive

- If $q<1$, the additional capital costs more than its market value: investment is unattractive

Expectations of high future profits raise the market value of capital and can increase investment today.


### Why does investment move so much over the business cycle?

Investment is much smaller than consumption, but it is considerably more volatile.

One reason is that investment changes the *stock* of capital:

- firms can postpone investment when demand is weak or uncertainty is high

- a relatively small change in the desired capital stock can imply a large percentage change in current investment


### Consumption, investment, and the real interest rate

The real interest rate connects household and firm decisions.

**Households:** The real interest rate changes the price of current consumption relative to future consumption.

**Firms:** The real interest rate is part of the user cost of capital and therefore affects desired investment.

These mechanisms will help explain why aggregate demand responds to monetary policy in the short-run model.


### What have we learned?

- Households use saving and borrowing to smooth consumption over time

- Expected lifetime income, current cash flow, access to credit, and real interest rates all affect consumption

- Firms compare the marginal product of capital with the user cost of capital

- Investment rises when expected profitability is high relative to financing, depreciation, and installation costs

- Consumption and investment provide the link from real interest rates to aggregate demand
