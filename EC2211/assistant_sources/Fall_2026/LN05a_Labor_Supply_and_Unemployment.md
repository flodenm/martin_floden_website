# Lecture Notes 5a: Labor Supply and Unemployment

**Course:** EC2211 Intermediate Macroeconomics, Stockholm University  
**Instructor:** Martin Flodén  
**Course version:** Fall 2026 (authoritative)

This is an agent-oriented rendering of `main_LN05a.tex`, checked against the 55-page `EC2211_Fall2026_LN05a.pdf`. Section and frame headings follow the lecture. Figure descriptions convey what the figures support; consult the lecture PDF for the visual itself. Data charts in the lecture are marked 2026 Q2 where stated.


## Introduction


### How is the Swedish labor market doing?

**Magdalena Andersson (S):** *“100 000 fler arbetslösa”*

sedan regeringsskiftet 2022

**Moderaterna:** *“110 000 fler sysselsatta”*

sedan 2022

**Can both statements be true?**


### Employment and unemployment

Yes. The labor force can grow, so the number of employed and the number of unemployed can rise at the same time.

The two rates also answer different questions:

- **Employment** asks how many people in the population have a job

- **Unemployment** asks how many people in the **labor force** are looking for a job

Employment and unemployment are not two complementary groups: some people are outside the labor force.

A country can have both high employment and high unemployment if many people participate in the labor market.


### Contents and literature

- Measuring labor-market outcomes

- Labor supply

- Labor demand and unemployment

Literature:

Jones (2024), chapter 7 and appendix to chapter 15

Steinsson (2025), [“Work and leisure”](https://eml.berkeley.edu/~jsteinsson/teaching/labor.pdf), sections 1–2 and 4–4.4


## Measuring labor-market outcomes


### Employment, unemployment and participation

**Employed:** persons who worked at least one hour for pay or profit during the reference week, or who were temporarily absent from a job because of holidays, sick leave, parental leave, etc.

**Unemployed:** persons who were not employed, but were available for work and actively searching for work

**Labor force:** employed plus unemployed persons

**Outside the labor force:** persons who were neither employed nor unemployed


### Rates with different denominators

Let $N=E+U+O$, where $O$ denotes people outside the labor force.

$$\begin{aligned}
        \text{Employment rate}       & = \frac{E}{N} \\[0.4em]
        \text{Participation rate}    & = \frac{E+U}{N}\\[0.4em]
        \text{Unemployment rate}     & = \frac{U}{E+U}
    
\end{aligned}$$

Comparisons also depend on

- the age range used to define the population

- whether people are studying

- the reference period and seasonal adjustment


### Employment rates

**Figure source:** `EmpRate.pdf`

**Figure description:** Bar chart comparing employment rates for ages 15–74 and 25–64 across Sweden, Denmark, Finland, Norway, France, Germany, Italy, Spain, and the EU. Age definitions matter because studying and retirement affect the broad-age rate.

Employment rates in 2026 Q2 for ages 15–74 and 25–64. Source: Eurostat LFS.


### Why does the age range matter?

- Many people aged 15–24 are in education rather than employment

- Many people aged 65–74 have retired

- Countries differ in the length of education and the effective retirement age

Employment rates for ages 15–74 therefore mix labor-market outcomes with differences in education and retirement.

**Always report the age range!**


### Employment rates for men and women

**Figure source:** `EmpRate_sex.pdf`

**Figure description:** Bar chart comparing male and female employment rates at ages 15–74 in the same nine geographic groups.

Employment rates in 2026 Q2 for ages 15–74. Source: Eurostat LFS.


### Employment rates for young and older people

**Figure source:** `EmpRate_youngold.pdf`

**Figure description:** Bar chart comparing employment rates at ages 15–19 and 65–74 in the same nine geographic groups.

Employment rates in 2026 Q2 for ages 15–19 and 65–74. Source: Eurostat LFS.


### Unemployment rates

**Figure source:** `UnempRate.pdf`

**Figure description:** Bar chart comparing unemployment rates for ages 15–74 and 25–64 in the same nine geographic groups. The denominator is the labor force within the age range.

Unemployment rates in 2026 Q2 for ages 15–74 and 25–64. Source: Eurostat LFS.


### Youth unemployment

**Figure source:** `UnempRate_young.pdf`

**Figure description:** Bar chart comparing unemployment rates for ages 15–19 and 20–24 in the same nine geographic groups. These rates refer to young people in the labor force, not all young people.

Unemployment rates in 2026 Q2 for ages 15–19 and 20–24. Source: Eurostat LFS.


### What does youth unemployment measure?

A student looking for a part-time job can be counted as unemployed.

The unemployment rate is $$\frac{\text{unemployed young people}}
        {\text{employed and unemployed young people}}$$ not the fraction of all young people who are unemployed.

Sweden has both a high employment rate and a high unemployment rate among young people. The prevalence of student employment and job search matters for both measures.


### Youth unemployment and NEET measure different outcomes

**Youth unemployment rate:** $$\frac{\text{unemployed young people}}
        {\text{young people in the labor force}}$$

**NEET rate:** $$\frac{\text{young people neither employed nor in education or training}}
        {\text{young population}}$$

A student searching for work may raise the youth unemployment rate without raising the NEET rate.


### Youth inactivity

**Figure source:** `NEET.pdf`

**Figure description:** Bar chart showing the share of people aged 15–24 neither employed nor in education or training (NEET) in the same nine geographic groups.

NEET rates in 2026 Q2 for ages 15–24. Source: Eurostat LFS.


### Unemployment by country of birth

**Figure source:** `Unemp2564_total.pdf`

**Figure description:** Country ranking of overall unemployment rates at ages 25–64.

**Figure source:** `Unemp2564_domestic.pdf`

**Figure description:** Country ranking of unemployment rates at ages 25–64 among people born domestically.

**Figure source:** `Unemp2564_foreign.pdf`

**Figure description:** Country ranking of unemployment rates at ages 25–64 among foreign-born people.

Unemployment rates in 2026 Q2 for ages 25–64. Source: Eurostat LFS.


### The Swedish labor market over time

**Figure source:** `SE_birth_timeseries.pdf`

**Figure description:** Two-panel Swedish time series of unemployment and employment rates at ages 25–64, comparing people born in Sweden with those born outside the EU27; observations extend to 2026 Q2.

Unemployment and employment rates for ages 25–64. Source: Eurostat LFS.


### The Swedish labor market has several descriptions

- Employment is high among people of prime working age

- Labor-force participation is high, particularly among women

- Unemployment is high, particularly among young and foreign-born people

- Youth unemployment partly reflects students searching for work

- The NEET rate gives a different picture of youth inactivity

These statements describe different outcomes. None provides a complete summary of the labor market.


## Labor supply


### Two margins of labor supply

Households make two conceptually different labor-supply decisions:

- **Extensive margin:** whether to participate in the labor market

- **Intensive margin:** how many hours to work when employed

The model in this section focuses on the intensive margin. Participation decisions concern the extensive margin, while employment also depends on labor demand.


### Temporary wage changes

Consider the following hypothetical offer:

I need research assistants for the coming two weeks. You can choose to work $H_1$ hours in the first week and $H_2$ hours in the second week.

- In the first week, I offer an hourly wage of EUR 15

- In the second week, I offer an hourly wage of EUR 50

How would you allocate your working hours between the two weeks?


### Permanent wage changes

Consider instead these two scenarios:

1.  Your hourly wage will be EUR 15 for the rest of your life

2.  Your hourly wage will be EUR 50 for the rest of your life

In both scenarios, you can choose how many hours to work per week.

Would you work more or less if your wage were permanently higher?


### Keynes predicted much shorter working hours

**Figure source:** `keynes.jpg`

**Figure description:** Portrait of John Maynard Keynes (1883–1946).

John Maynard Keynes  
1883–1946

In [“Economic possibilities for our grandchildren”](http://www.econ.yale.edu/smith/econ116a/keynes1.pdf), Keynes (1930) predicted that

- living standards in progressive countries would become four to eight times higher over the following century, and

- a fifteen-hour work week might become sufficient

Rising productivity has raised consumption substantially. It has also reduced working hours, but much less than Keynes anticipated.


### Hours worked per working-age person, USA

**Figure source:** `Steinsson_fig4.jpg`

**Figure description:** Annual hours worked per working-age person in the United States, about 1950–2020. The line fluctuates around roughly 1,200–1,400 hours with no sustained long-run downward trend in this period. Figure 4 in Steinsson (2025).

Figure 4 in Steinsson (2025)


### Hours worked, 1870--1998

**Figure source:** `BoppartKrusell_fig1.jpg`

**Figure description:** Boppart and Krusell (2020), Figure 1: average yearly hours worked per capita in 25 countries at selected dates from 1870 to 1998. The lines generally fall strongly across the twentieth century, though levels and timing differ across countries. Source in figure: Maddison (2001). The image was found in the archived course zip, not among the current LN5a attachments.

[Boppart and Krusell (2020)](https://www.journals.uchicago.edu/doi/full/10.1086/704071)


### Temporary and permanent wage changes

A temporary wage increase and a permanent wage increase may have very different effects on labor supply.

A temporarily higher wage makes working now particularly attractive relative to working at another time.

A permanent wage increase also makes the household richer. This may increase its demand for leisure and reduce hours worked.

Economic theory separates these two forces into a **substitution effect** and an **income effect**.


### Preferences and the budget constraint

Suppose that a household values consumption, $C$, and dislikes hours worked, $H$: $$U(C)-V(H),$$ where $U'(C)>0$ and $V'(H)>0$.

In a one-period model without saving, the budget constraint is $$C=wH+T,$$ where $w$ is the hourly wage and $T$ is non-labor income.


### The household's choice

The household chooses consumption and hours worked: $$\max_{C,H}\left\{U(C)-V(H)\right\}
        \qquad \text{subject to} \qquad C=wH+T$$

Use the budget constraint to substitute for consumption: $$\max_H\left\{U(wH+T)-V(H)\right\}$$

The first-order condition is $$wU'(C)=V'(H)$$


### Interpreting the labor-supply condition

$$\underbrace{wU'(C)}_{\text{benefit from one more hour}}
        =
        \underbrace{V'(H)}_{\text{cost of one more hour}}$$

- One additional hour generates $w$ units of consumption

- Each unit of consumption raises utility by $U'(C)$

- The additional hour also raises the disutility of work by $V'(H)$

If the benefit exceeded the cost, the household would want to work more.


### A wage increase has two effects

Suppose that $$U(C)=\alpha\ln C,
        \qquad
        V(H)=-(1-\alpha)\ln(1-H),$$ where total available time is normalized to one and $1-H$ is leisure.

With $T=0$, the budget constraint is $C=wH$. The first-order condition implies $$H=\alpha,
        \qquad
        C=\alpha w.$$

Labor supply does not depend on the wage in this particular example.


### Income and substitution effects

A higher wage has two opposing effects on labor supply:

- **Substitution effect:** leisure becomes more expensive, so the household wants to work more

- **Income effect:** the household becomes richer and wants more leisure, so it wants to work less

With the logarithmic preferences on the previous slide and no non-labor income, the two effects cancel exactly.

This exact cancellation is a property of the example, not a general result.


### Two interpretations of labor-supply responses

- **Across time:** workers may shift hours toward periods when their wage is temporarily high

- **Across countries or over long horizons:** persistently higher wages also make households richer and increase their demand for leisure

Short-run and long-run labor-supply responses therefore need not be the same.

Responses may also differ between hours per worker and the decision to participate.


### Income taxation

With an income tax $\tau$, the household’s budget constraint is $$C=(1-\tau)wH+T.$$

Consider two cases:

1.  The tax revenue does not return to the household: $T=0$

2.  The tax revenue is returned as a lump-sum transfer: $$T=\tau w\bar H,$$ where the individual household takes average hours $\bar H$ as given


### Taxes, income effects, and substitution effects

**Case 1:** With $T=0$, the tax reduces the net wage and disposable income:

- the substitution effect reduces hours worked,

- the income effect increases hours worked, and

- the two effects cancel under our logarithmic specification.

**Case 2:** With a lump-sum transfer, the tax reduces the return to an additional hour without reducing household income to the same extent. The substitution effect then dominates and labor supply falls.

This distinction will be important when we compare working hours in Europe and the United States.


### Why do Europeans work fewer hours than Americans?

**Figure source:** `MACRO6_Table07.02.jpg`

**Figure description:** Jones (2024), Table 7.2: hours worked per person, indexed so the United States equals 100 in 2019. In 1960, 1990, 2019 respectively: United States 88, 103, 100; Italy 107, 87, 86; France 111, 79, 75; Germany 131, 93, 88; U.K. 114, 96, 96; Japan 130, 125, 110; South Korea 89, 134, 122. Source: Penn World Table 10.0.


### Taxes and the Europe--U.S. hours gap

Prescott (2004) argued that higher taxes could explain why Europeans work less than Americans.

- A higher tax wedge reduces how much consumption an additional hour of work can finance.

- But a permanent tax wedge need not reduce labor supply: the substitution and income effects work in opposite directions.

- **How the tax revenue is used is crucial.** Transfers and public services that benefit households limit the loss of income. The income effect is then weaker, and the substitution effect toward leisure becomes more important.

Other proposed explanations include preferences and social interactions, labor-market institutions, and differences in the composition of employment.

Further reading: [Prescott (2004)](https://www.minneapolisfed.org/research/qr/qr2811.pdf); [Olovsson (2009)](https://doi.org/10.1111/j.1468-2354.2008.00523.x) for an analysis focusing on Sweden; and [Steinsson, Sections 4.3–4.4](https://eml.berkeley.edu/~jsteinsson/teaching/labor.pdf).


## Labor demand and unemployment


### Labor supply and labor demand

**Labor demand**

- Firms choose labor input to maximize profits

- Labor demand falls with the wage because the marginal product of labor is decreasing

**Labor supply**

- Households choose whether to work and how many hours to supply

- Income and substitution effects determine the response to wages

The equilibrium wage is such that labor demand equals labor supply.


### Labor supply and labor demand

**Figure source:** `MACRO6_FIG07.03.jpg`

**Figure description:** Jones (2024), Figure 7.3: wage on the vertical axis and employment on the horizontal axis. An upward-sloping labor-supply curve and downward-sloping labor-demand curve intersect at equilibrium wage and employment.

Figure 7.3 in Jones (2024).


### Unemployment concepts

**Natural rate of unemployment:** unemployment that remains when the economy is neither in a boom nor in a recession

**Frictional unemployment:** unemployment while workers search and move between jobs

**Structural unemployment:** unemployment associated with mismatch and labor-market institutions

**Cyclical unemployment:** unemployment associated with fluctuations in aggregate economic activity


### Unemployment over time

**Figure source:** `unemp_SE_US.png`

**Figure description:** OECD Economic Outlook time series of unemployment rates at ages 15–74, Sweden and United States, 1960–2024. Swedish unemployment is low before the early-1990s crisis and rises sharply thereafter; U.S. unemployment is more cyclical, with a marked spike in 2020.

Unemployment rates for ages 15–74. Source: OECD Economic Outlook.


### Participation and employment over time

**Figure source:** `EmpPart_SE_US.png`

**Figure description:** OECD Economic Outlook time series of labor-force participation (solid) and employment (dashed) at ages 15–74, Sweden and United States, 1960–2024. Participation and employment are generally higher in Sweden than in the United States in recent decades. The gap between participation and employment reflects unemployment within the labor force.

Labor-force participation and employment rates for ages 15–74. Source: OECD Economic Outlook.


### Questions raised by the time series

- Why was Swedish unemployment so low before the crisis of the early 1990s?

- Why is US unemployment more volatile over the business cycle?

- Why did US unemployment rise much more sharply in 2020?

- Why is labor-force participation higher in Sweden?

Different institutions can affect unemployment, participation and hours worked in different ways.


### A reduction in labor demand with wage rigidity

**Figure source:** `MACRO6_FIG07.06.jpg`

**Figure description:** Jones (2024), Figure 7.6: after labor demand shifts left, the flexible-wage equilibrium would move from A to B, with lower wage and employment. With the wage fixed at the initial level, employment falls further to point C on the new demand curve, creating unemployment relative to labor supply at that wage.

Figure 7.6 in Jones (2024).


### The bathtub model of unemployment

The model describes flows between employment and unemployment.

Notation:

- $L$: labor force

- $E_t$: number of employed people

- $U_t$: number of unemployed people

- $s$: job-separation rate

- $f$: job-finding rate

The labor force is fixed and satisfies $$L=E_t+U_t.$$


### Flows into and out of unemployment

Each period,

- $sE_t$ employed workers lose or leave their jobs, and

- $fU_t$ unemployed workers find jobs

**Flow diagram:** Employed workers move to unemployment at rate $sE_t$; unemployed workers move to employment at rate $fU_t$.

The number of unemployed people therefore changes according to $$U_{t+1}-U_t=sE_t-fU_t$$

Unemployment rises when inflows exceed outflows, etc.


### Steady-state unemployment

In a steady state, unemployment is constant: $$sE^*=fU^*$$

Use $E^*=L-U^*$ to obtain $$s(L-U^*)=fU^*$$

The steady-state unemployment rate is therefore $$u^*\equiv\frac{U^*}{L}=\frac{s}{s+f}$$

A higher separation rate raises unemployment. A higher job-finding rate reduces it.


### The fixed labor force limits the basic model

The basic model implies $$u^*=\frac{s}{s+f}.$$ The unemployment rate is determined by the rates of job separation and job finding.

But the model also assumes $$E_t=L-U_t,$$ where $L$ is fixed. Employment and unemployment must therefore move in opposite directions.

The model cannot explain

- changes in labor-force participation, or

- an increase in both employment and unemployment

To study these outcomes, we allow people to enter and leave the labor force.


### An extended bathtub model

The population, $N$, is fixed and divided into three groups: $$N=\underbrace{E_t+U_t}_{\text{labor force}}+D_t$$ where $D_t$ denotes discouraged people **outside the labor force**.

**Flow diagram:** Employment $E_t$ and unemployment $U_t$ exchange flows $sE_t$ and $fU_t$; unemployment $U_t$ and outside the labor force $D_t$ exchange flows $dU_t$ and $aD_t$.

Unemployed workers become discouraged at rate $d$. People outside the labor force become active and start searching at rate $a$.


### Steady state in the extended model

In a steady state, the flows between each pair of groups balance: $$sE^*=fU^*,
        \qquad
        dU^*=aD^*.$$

Combining the flow conditions with $N=E^*+U^*+D^*$ gives $$\textbf{employment rate} = \frac{E^*}{N}=\frac{af}{a(s+f)+sd}$$


### Activation, employment, and unemployment

A higher activation rate $a$ moves people from outside the labor force into unemployment.

- The number unemployed rises

- More unemployed people subsequently find jobs, so employment rises

- The employment and participation rates therefore rise

Within the labor force, steady-state flows still satisfy $$sE^*=fU^*$$ The unemployment rate is therefore $$u^*\equiv\frac{U^*}{E^*+U^*}
        =\frac{s}{s+f},$$ which does not depend on $a$ or $d$.


### What does the extended model explain?

Return to the statements at the beginning of the lecture:

*More people are employed, but more people are also unemployed.*

In the extended model, a higher activation rate $a$

- reduces the number of discouraged people outside the labor force

- raises the number unemployed

- raises employment as some of the newly active workers find jobs

The employment rate rises, but $$u^*=\frac{U^*}{E^*+U^*}=\frac{s}{s+f}$$ remains unchanged.

**The model can explain why employment and unemployment increase simultaneously, but not why both rates increase.**


### Heterogeneity can make both rates rise

Suppose that employment consists of two groups: $$E_t=\bar E_H+E_{L,t}.$$

Workers in $\bar E_H$ have high labor-market attachment and are always employed. Workers in $E_L$, $U$, and $D$ move as in the extended model.

A higher activation rate moves more people into the group with lower labor-market attachment. Some find jobs, while others remain unemployed.

- The employment rate rises

- The unemployment rate also rises

Many Swedish labor-market policies aim to activate people who are not employed. Their effects cannot be evaluated using the unemployment rate alone.
