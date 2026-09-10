# Lecture Notes 4: Growth Beyond the Solow Model

**Course:** EC2211 Intermediate Macroeconomics, Stockholm University  
**Instructor:** Martin Flodén  
**Course version:** Fall 2026 (authoritative)

This file is an agent-oriented rendering of the final Fall 2026 lecture source. Headings reproduce section and frame titles. Figure notes and nearby text should be used when answering questions about visuals.

## Introduction

### Growth is a recent phenomenon

**Figure source:** `MACRO6_FIG03.01.jpg`

**Figure description:** Very long-run GDP per person for the United States, Japan, the United Kingdom, Argentina, China, and Ghana. Income per person remains near low levels for most of history and rises sharply only in the modern era, with large differences in timing and magnitude across countries.

### Can a supercomputer generate sustained growth?

In Lecture 2, we considered the EU’s new AI-focused supercomputer in Finland. It combines

- **physical capital**: processors, servers, storage, and buildings

- **human capital**: the skills needed to build, operate, and use the system

- **ideas**: algorithms, scientific knowledge, and new ways of organizing production.

The Solow model explains the accumulation of physical and human capital. But it treats the growth of ideas—TFP growth—as exogenous.

### Why did sustained growth begin?

For most of human history, improvements in technology did not produce sustained growth in living standards.

We now ask

- Why did productivity improvements raise population rather than income per person before the Industrial Revolution?

- Why can the accumulation of ideas generate sustained growth?

- Why do some societies provide stronger incentives to create and adopt new ideas?

- Why does innovation create both gains and disruption?

### Contents and literature

- Malthus and the transition to sustained growth

- Ideas and the Romer model

- Institutions and growth

- Innovation and creative destruction

Literature:

- Jones (2024), chapter 6

- The Royal Swedish Academy of Sciences (2018), [“Integrating nature and knowledge into economics”](https://www.nobelprize.org/prizes/economic-sciences/2018/popular-information/), in particular pages 1–4

- The Royal Swedish Academy of Sciences (2024), [“They provided an explanation for why some countries are rich and others poor”](https://www.nobelprize.org/prizes/economic-sciences/2024/popular-information/)

- The Royal Swedish Academy of Sciences (2025), [“From stagnation to sustained growth”](https://www.nobelprize.org/prizes/economic-sciences/2025/popular-information/)

## Malthus and the transition to sustained growth

### The Malthusian mechanism

Malthus combined

- land in fixed supply

- diminishing marginal product of labor, and

- population growth that rises with income.

Higher productivity initially raises income per person. The resulting population growth then reduces land per person and pushes income back down.

**Figure source:** `malthus.jpg`

**Figure description:** Portrait of Thomas Malthus (1766–1834).

Thomas Malthus  
1766–1834

### Production and population

The Malthusian production function is $$
        Y_t=A_tD^\lambda L_t^{1-\lambda},$$ where land, $D$, is in fixed supply. Output per person is therefore $$
        y_t\equiv\frac{Y_t}{L_t}
        =A_t\left(\frac{D}{L_t}\right)^\lambda.$$

Population growth is endogenous: $$
        \frac{L_{t+1}}{L_t}=\frac{y_t}{y^s},$$ where $y^s$ is the subsistence level.

Population grows when $y_t>y^s$ and shrinks when $y_t<y^s$.

### A productivity improvement raises population in the long run

Consider a permanent increase in $A$.

1.  Output per person and wages initially rise.

2.  Because $y_t>y^s$, the population grows.

3.  Land per person, $D/L_t$, falls.

4.  Diminishing marginal product of labor pushes output per person back toward $y^s$.

Long-run implication The productivity improvement supports a larger population, but not a permanently higher level of income per person.

### Even continuing technological progress need not raise income

Suppose that productivity grows at the constant rate $g_A$. Applying the growth-rate rules to equation (eq:MalthusPerCapita) gives $$g_y=g_A-\lambda g_L.$$

Along a balanced growth path, equation (eq:MalthusPopulation) implies that the population growth factor – and therefore income per person – must be constant: $$g_y=0.$$ It follows that $$g_L=\frac{g_A}{\lambda}.$$

In the Malthusian model, technological progress generates population growth rather than sustained growth in income per person.

### The Black Death illustrates the Malthusian mechanism

**Figure source:** `CORE_Malthus_figure-01-12-c.png`

**Figure description:** English real-wage index plotted against population for observations from the 1280s through the 1610s. The Black Death reduced population sharply and real wages rose; as population later recovered, real wages fell. The inverse movement illustrates the Malthusian mechanism operating through land per worker.

- The plague sharply reduced the population in the fourteenth century, and real wages rose.

- Population subsequently recovered and real wages fell.

- By 1610, both were close to their levels in 1280.

Material from [COREecon: The Economy 2.0](https://www.core-econ.org/the-economy/microeconomics/01-prosperity-inequality-07-malthusian-trap.html#nav)

### Real wages in England eventually began to rise

**Figure source:** `MalthusWages.png`

**Figure description:** Real wages of laborers in England from 1250 to 2000 on a ratio scale. Wages fluctuate without a persistent upward trend for centuries, then begin a sustained rise around the Industrial Revolution and accelerate strongly after about 1850.

Figure 1 in [Steinsson (2025), "Malthus and pre-industrial stagnation"](https://jonsteinsson.com/teaching/malthus.pdf)

### Why did economies escape the Malthusian trap?

The Malthusian mechanism weakened when

- technological progress became faster and more persistent

- land became less important relative to capital and knowledge

- fertility stopped rising with income, and eventually declined

- scientific and practical knowledge began to accumulate systematically.

The Industrial Revolution was therefore not simply the arrival of one important invention. It marked a transition to **sustained and cumulative innovation**.

### Mokyr: useful knowledge must be created and maintained

Joel Mokyr emphasizes that sustained technological progress requires more than new ideas:

- scientific understanding that helps generate further advances

- engineers and skilled craftspeople who can implement, improve, and maintain new technologies

- institutions and norms that permit experimentation and the exchange of knowledge

- openness to innovations that may threaten established interests.

This helps explain why technological progress became self-sustaining after the Industrial Revolution but had repeatedly faded away before it.

## Ideas and the Romer model

### The Solow model does not explain technological progress

In the Solow model, $$Y_t=A_tK_t^\alpha L_t^{1-\alpha}.$$

- Capital accumulation cannot generate sustained growth in output per worker because the marginal product of capital diminishes

- Sustained growth requires growth in $A_t$

- But the path of $A_t$ is imposed from outside the model

We now make technological progress endogenous by allowing workers to produce new ideas.

### Romer made the production of ideas part of the model

- Paul Romer developed models in which firms and researchers deliberately create new ideas

- The incentive to innovate depends on the expected economic return to a successful idea

- Romer received the 2018 prize “for integrating technological innovations into long-run macroeconomic analysis”

**Figure source:** `romer.jpg`

**Figure description:** Portrait of Paul Romer (born 1955), associated with the theory of idea-driven endogenous growth and the 2018 Nobel Memorial Prize in Economic Sciences.

Paul Romer  
1955–

### Objects are rival; ideas are non-rival

**Objects**

- A machine used by one firm cannot simultaneously be used by another firm

- Dividing a fixed stock of capital among more workers reduces capital per worker

**Ideas**

- The same formula, design, or algorithm can be used by many producers at the same time

- Using an idea does not reduce the amount available to anyone else

Non-rivalry allows the benefits from an idea to grow with the scale of the economy.

### Non-rivalry is the key distinction

**Figure source:** `Nonrivalry goods.jpg`

**Figure description:** Classification of rival and non-rival goods by degree of control or excludability. Rival examples include a soft drink, a television, and fish in the ocean. Non-rival examples include a coded satellite broadcast, a secret recipe, software, a GPS signal, basic R&D, and the Pythagorean theorem. Non-rivalry and excludability are separate properties.

From [The Royal Swedish Academy of Sciences (2018)](https://www.nobelprize.org/prizes/economic-sciences/2018/popular-information/)

### Ideas can be non-rival but partially excludable

Non-rivalry does not mean that everyone automatically receives free access to every idea:

- Patents, copyright, secrecy, and technological complexity can allow an innovator to restrict use

- Some excludability allows innovators to earn a return on costly research

- But restricting access also prevents some socially valuable uses of an idea

Innovation policy therefore confronts a trade-off between **incentives to create ideas** and **diffusion of existing ideas**.

### Why competitive factor pricing is no longer enough

In Lecture 2, constant returns to capital and labor implied $$rK+wL=\alpha Y+(1-\alpha)Y=Y.$$

All output was used to pay capital and labor. But suppose production also relies on an idea that was costly to create.

- Non-rival ideas introduce increasing returns: the idea is created once and then used at many production sites

- Perfect competition would leave no revenue with which to pay for creating the idea

- Temporary market power can provide the required reward

Romer’s full theory therefore links ideas, increasing returns, and imperfect competition.

### The Romer model: basic structure

We use the simplified Romer model as presented in Jones’ textbook:

- The economy produces a consumption good and new ideas

- Workers can produce either goods or ideas

- The existing stock of knowledge makes workers in goods production more productive

- Physical capital is omitted to isolate the role of ideas

- The allocation of labor between the two activities is taken as given

### Notation

- $Y_t$: output of the consumption good

- $A_t$: stock of knowledge

- $L_t$: population

- $n$: population growth rate

- $L_{yt}$: workers producing goods

<!-- -->

- $L_{at}$: workers producing ideas

- $\gamma$: elasticity of output with respect to knowledge

- $\lambda$: fraction of workers producing ideas

- $\zeta$: research productivity

- $A_0$: initial stock of knowledge

(Note that $\lambda$ had a different interpretation in the Malthusian model earlier in these slides.)

### Knowledge raises output in goods production

Production of the consumption good is $$
        Y_t=A_t^\gamma L_{yt}.$$

- $A_t$ is available to every worker in goods production

- A larger stock of knowledge therefore raises output per worker

- The parameter $\gamma>0$ determines how strongly knowledge affects production

Because knowledge is non-rival, the same $A_t$ can raise the productivity of all $L_{yt}$ workers simultaneously.

### Researchers produce new ideas

Production of new ideas is $$
        \Delta A_{t+1}=\zeta L_{at}.$$

The labor resource constraint is $$
        L_{yt}+L_{at}=L_t,$$ and population evolves according to $$
        L_{t+1}=(1+n)L_t.$$

A constant fraction $\lambda$ of the population works in research: $$
        L_{at}=\lambda L_t,
        \qquad
        L_{yt}=(1-\lambda)L_t.$$

### Output per person depends on the stock of knowledge

Divide equation (eq:RomerProduction) by $L_t$ and use $L_{yt}=(1-\lambda)L_t$: $$
        y_t\equiv\frac{Y_t}{L_t}
        =A_t^\gamma(1-\lambda).$$

Output per person is therefore increasing in the stock of knowledge, $A_t$.

Compare this with the Solow model:

- Because physical capital is *rival*, output per person depends on capital intensity, $k_t=K_t/L_t$.

- Because knowledge is *non-rival*, the stock $A_t$ does not have to be divided among workers. The same knowledge can raise the productivity of all workers.

Sustained growth in $y_t$ therefore requires continued accumulation of the non-rival stock of knowledge, $A_t$.

### The growth rate of knowledge depends on research effort

Divide equation (eq:RomerIdeas) by $A_t$: $$
        g_{At}\equiv\frac{\Delta A_{t+1}}{A_t}
        =\zeta\frac{L_{at}}{A_t}
        =\zeta\lambda\frac{L_t}{A_t}.$$

- More researchers raise the flow of new ideas

- A larger existing stock of knowledge makes any given flow of new ideas smaller relative to the stock

- A constant growth rate therefore requires researchers and knowledge to grow at the same rate

### Knowledge and population grow together

On a balanced growth path, $g_A$ is constant. Rearranging equation (eq:RomerKnowledgeGrowth) gives $$A_t=\frac{\zeta\lambda}{g_A}L_t.$$

Thus, $A_t$ is proportional to $L_t$. Because population grows at rate $n$, the stock of knowledge must also grow at rate $n$: $$
        g_A=n.$$

Population growth continually increases the number of researchers. Their new ideas continually expand the stock of knowledge.

### Ideas generate sustained growth in output per person

From equation (eq:RomerPerCapita), $$y_t=A_t^\gamma(1-\lambda).$$ Since $1-\lambda$ is constant, the growth-rate rules give $$g_y=\gamma g_A.$$ Using $g_A=n$ on the balanced growth path, $$
        \boxed{g_y=\gamma n.}$$

Unlike physical capital, knowledge does not become less useful when it is shared among more workers. Its accumulation can therefore sustain growth in output per person.

### Higher research productivity raises growth temporarily

Suppose that research productivity $\zeta$ rises permanently. From equation (eq:RomerKnowledgeGrowth), $$g_{At}=\zeta\lambda\frac{L_t}{A_t}.$$

- The growth rate of knowledge rises immediately

- Knowledge and output per person move toward higher paths

- As $A_t/L_t$ rises, the growth rate of knowledge falls back

- Long-run growth remains $g_A=n$ and $g_y=\gamma n$

A permanent increase in $\zeta$ has a permanent **level effect**, but only a temporary **growth effect**.

### A permanent rise in research productivity

**Figure source:** `MACRO6_FIG06.03.jpg`

**Figure description:** Output per person over time after a permanent increase in research productivity. Output does not fall on impact; growth temporarily accelerates and the economy converges to a higher parallel balanced-growth path. The long-run growth rate is unchanged, but there is a permanent positive level effect.

Figure 6.3 in Jones (2024). His $\overline{z}$ is our $\zeta$.

### A larger research share creates a trade-off

Suppose that the fraction of workers producing ideas, $\lambda$, rises.

- **Immediate cost:** fewer workers produce consumption goods, so the factor $1-\lambda$ falls

- **Dynamic benefit:** more researchers produce more ideas, so knowledge grows faster during the transition

- **Long run:** the economy reaches a higher knowledge path if the additional knowledge outweighs the loss of goods-producing workers

- The long-run growth rate nevertheless returns to $g_y=\gamma n$

A higher research share does not provide a free lunch: research uses labor that could have produced goods today.

### A permanent rise in the research share

**Figure source:** `MACRO6_FIG06.04.jpg`

**Figure description:** Output per person over time after a permanent increase in the research share. Moving labor from goods production to research causes an immediate downward jump in output, followed by faster transitional growth and convergence to a higher parallel balanced-growth path. Whether output is higher at a particular date depends on the initial cost and later knowledge gain.

Figure 6.4 in Jones (2024). His $\overline{l}$ is our $\lambda$.

### Population has different roles in the three models

| **Model** | **Role of population growth**                                                                                     |
|:----------|:------------------------------------------------------------------------------------------------------------------|
| Malthus   | Productivity gains support a larger population rather than sustained growth in income per person.                 |
| Solow     | Faster population growth requires more investment to equip new workers and lowers steady-state output per worker. |
| Romer     | Faster population growth eventually produces more researchers and faster growth in the stock of ideas.            |

The models therefore give very different interpretations of the global decline in fertility.

### Total fertility rates have trended down

**Figure sources:** `TFR_a.png` and `TFR_b.png`

**Figure description:** The first chart shows total fertility rates from 1960 to the early 2020s for Sweden, the United States, Germany, Italy, and Japan. Fertility generally declines and is below replacement near the end, with especially low rates in Italy and Japan and cyclical variation in Sweden and the United States. The second chart shows Korea, China, Nigeria, India, and Egypt. Fertility declines in every country, extraordinarily sharply in Korea and China; Nigeria remains much higher but also trends downward.

Total fertility rates (the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with age-specific fertility rates of the specified year). Source: [World Bank](https://data360.worldbank.org/en/indicator/WB_WDI_SP_DYN_TFRT_IN?view=datatable&recentYear=false)

### The model isolates one mechanism

The simplified Romer model makes the non-rivalry of ideas transparent. It does not explain everything about innovation:

- The research share $\lambda$ is imposed rather than chosen.

- Every researcher produces the same flow of ideas, regardless of the existing stock of knowledge.

- The model does not describe firms, profits, patents, or competition explicitly.

- Population growth is the only determinant of long-run growth.

To understand why research is undertaken and why innovations are adopted, we need to examine incentives and institutions.

## Institutions and growth

### Proximate versus fundamental causes

The factors we have listed (innovation, economies of scale, education, capital accumulation, etc.) are not causes of growth; they are growth.

Physical capital, human capital, and ideas are **proximate determinants** of income.

A deeper question is why societies create different incentives to invest, acquire skills, innovate, and adopt new technologies.

### Institutions shape economic incentives

Institutions are the rules that organize political and economic activity.

Relevant examples include:

- protection of property and intellectual-property rights

- enforcement of contracts

- constraints on political leaders

- the ability of new firms to enter markets

- the distribution of political power

Institutions influence both the return to innovation and whether those threatened by innovation can block it.

### Institutions help explain differences in prosperity

Daron Acemoglu, Simon Johnson, and James Robinson received the 2024 Nobel prize “for studies of how institutions are formed and affect prosperity.”

Their research addresses two difficult questions:

- How can we distinguish the effect of institutions from the effects of geography, culture, and income itself?

- If inclusive institutions promote prosperity, why are they not adopted everywhere?

### The reversal of fortune

Some of the relatively prosperous parts of the world in 1500 are relatively poor today, while some previously less prosperous regions are now rich.

Acemoglu, Johnson, and Robinson argue that European colonization contributed to this reversal:

- Densely populated and prosperous colonies often received extractive institutions designed to transfer resources

- Where Europeans settled in large numbers, they had stronger incentives to establish institutions protecting settlers’ property and political rights

- Many of these institutional differences persisted

### The empirical strategy seeks a source of institutional variation

The proposed causal chain is $$\begin{array}{c}
        \text{conditions facing European settlers}
        \\
        \Downarrow
        \\
        \text{colonial settlement and institutions}
        \\
        \Downarrow
        \\
        \text{persistent institutions}
        \\
        \Downarrow
        \\
        \text{income per person today}
        \end{array}$$

The historical source of institutional variation helps address the concern that rich countries may simply be better able to afford good institutions.

### Why can inefficient institutions persist?

Institutions that reduce total output may nevertheless benefit groups that hold political power:

- Political elites may receive large private gains from extractive institutions

- They may also fear losing power under more inclusive institutions

- A promise to introduce reforms later may not be credible: once an opposing group gives up its leverage, the elite may renege

Institutional change is therefore not merely a technical question of identifying policies that would raise aggregate income.

### What should we conclude from the institutional evidence?

- Institutions are an important cause of persistent differences in prosperity

- Political and economic institutions interact: political power affects economic rules, and economic outcomes affect political power

- Historical institutions can continue to shape outcomes long after the conditions that created them have disappeared

But the evidence does not imply that every institutional reform succeeds, or that one institutional feature mechanically determines income.

## Innovation and creative destruction

### The 2025 prize focused on innovation-driven growth

The 2025 Nobel prize was awarded

- to Joel Mokyr for identifying prerequisites for sustained growth through technological progress, and

- to Philippe Aghion and Peter Howitt for the theory of sustained growth through creative destruction.

Romer explains why non-rival ideas can sustain growth.

Aghion and Howitt focus on the process through which firms create better technologies that replace existing ones.

### Innovation is both creative and destructive

In the Aghion–Howitt framework,

1.  firms invest in research in the hope of discovering a better technology

2.  a successful innovator earns temporary monopoly profits

3.  the new technology raises productivity

4.  it also reduces the value of the previous technology and the firms using it

5.  later innovations eventually replace the current innovator.

Growth results from a continuing sequence of quality-improving innovations, not from smooth improvement within an unchanged set of firms.

### Romer and Aghion–Howitt emphasize different margins

|                   | **Romer**                                                | **Aghion–Howitt**                                     |
|:------------------|:---------------------------------------------------------|:------------------------------------------------------|
| Innovation        | New ideas expand productive possibilities                | Better technologies replace older technologies        |
| Private reward    | Market power allows innovators to recover research costs | Successful innovators earn profits until displaced    |
| Central mechanism | Accumulation of non-rival knowledge                      | Entry, firm turnover, and creative destruction        |
| Main concern      | Knowledge may be underprovided because it spills over    | Incumbents and innovators affect each other’s profits |

These are complementary perspectives on how innovation can sustain growth.

### Innovation requires both rewards and competitive pressure

Innovation creates several competing effects:

- **Knowledge spillovers:** innovators do not capture all the social benefits of their discoveries

- **Temporary market power:** expected profits provide an incentive to undertake costly and uncertain research

- **Business stealing:** innovators may earn profits partly by reducing the value of incumbent firms

- **Entrenched incumbents:** established firms may use economic or political power to prevent entry and diffusion

Institutions must reward successful experimentation while allowing new firms and technologies to challenge existing producers.

Patent policy, R&D subsidies, competition policy, openness, and social insurance all affect this balance. More of each is not always better.

## Growth at the technological frontier

### Growth is different at the technological frontier

Countries below the technological frontier can grow by

- accumulating physical and human capital

- adopting technologies already used elsewhere

- improving institutions and the allocation of resources

These opportunities become smaller as a country approaches the frontier.

At the frontier, sustained growth ultimately requires the creation of **new ideas and technologies**.

### Can growth at the frontier continue at its historical rate?

U.S. GDP per person has grown by approximately 2 percent per year over the past 150 years.

But the stability of historical growth does not guarantee that it will continue:

- Which forces generated the historical growth rate?

- Which of these forces can continue indefinitely?

- Could new technologies provide new sources of growth?

Jones (2023) uses growth accounting to examine these questions.

### Much of past U.S. growth came from forces that cannot rise forever

**Figure source:** `JonesGrowthAccounting.png`

**Figure description:** Stylized accounting for U.S. GDP-per-person growth since the 1950s. Of 2 percent annual growth, about 1.3 percentage points come from TFP, 0.5 from human capital per person, and 0.2 from the employment-population ratio. The TFP component is further divided into roughly 0.7 percentage points from research intensity, 0.3 from population growth, and 0.3 from misallocation. Several contributors cannot rise indefinitely.

- Measured TFP accounts for 1.3 percentage points of the historical 2 percent annual growth in GDP per person

- TFP growth in turn reflects research intensity, improved allocation of talent, and population growth

Figure 5 in [Jones (2023), “The Outlook for Long-Term Economic Growth.”](https://www.kansascityfed.org/Jackson%20Hole/documents/9771/GrowthOutlookPanel.pdf)

### The historical growth rate may overstate sustainable growth

Jones’s accounting attributes 1.7 of the historical 2 percentage points of growth to

- rising educational attainment

- a rising employment–population ratio

- rising research expenditure relative to GDP

- improved allocation of talent

These changes can raise growth for many decades, but each is bounded.

The remaining 0.3 percentage points are attributed to population growth, which determines the long-run growth of research effort in the semi-endogenous growth model.

### Future frontier growth faces both headwinds and tailwinds

**Possible headwinds**

- slower population growth

- stagnating educational attainment

- limits to the share of GDP devoted to research

- declining research productivity

- geopolitical fragmentation

**Possible tailwinds**

- better allocation of talent

- more researchers in emerging economies

- improved institutions and knowledge diffusion

- artificial intelligence

Growth may slow, but the outcome is not predetermined.

### AI may affect both productivity levels and long-run growth

Recall the production of new knowledge in the Romer model: $$\Delta A_{t+1}=\zeta L_{at}.$$

**AI in goods production**

AI can automate tasks and help workers use existing knowledge more effectively

This raises productivity and may generate faster growth while the technology is adopted

**AI in idea production**

AI may make researchers more productive, corresponding to a higher $\zeta$

It may eventually allow machines themselves to search for new ideas

- A one-time increase in $\zeta$ raises the level of output but does not permanently raise its growth rate in the simple Romer model

- AI raises long-run growth only if research productivity continues to improve or effective research input grows independently of population

### Return to the European AI supercomputer

The new supercomputer increases Europe’s physical capital and computing capacity.

Its longer-run effect depends on whether

- researchers use it to create new ideas

- firms adopt those ideas throughout the economy

- new firms can enter and challenge established producers

- workers acquire complementary skills

- institutions balance incentives to innovate against the diffusion of knowledge

### What we did

- Used the Malthusian model to explain preindustrial stagnation

- Explained why the accumulation of rival capital cannot by itself sustain growth

- Showed how non-rival ideas can generate sustained growth in the Romer model

- Examined institutions as a fundamental cause of prosperity

- Explained innovation as a process of creative destruction

- Considered whether historical frontier growth can continue and how AI might affect it

Main lesson Sustained growth requires more than accumulating objects. It depends on the creation, diffusion, and adoption of ideas – and on institutions that make this process possible.
