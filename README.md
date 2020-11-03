# Algorithmic-trading Library in Python

The AT Library is a python library that can be used to create trading algorithms using technical indicators. It's built on Pandas, Numpy, and Matplotlib.

![Example Chart](/images/sign_bb.png)

# Technical Indicators

### Trend
   * Simple Moving Average
   * Exponentiel Moving Average
   * Moving Average Convergence Divergence
### Momentum
   * Money Flow Index
   * Relative Strength Index
   * Stochastic Oscillator
   * Williams %R
   * Rate of Change
   * Chaikin Oscillator
### Volume
   * On Balance Volume
   * Negative Volume Index
   * Positive volume index
### Volatility
   * Bollinger Bands
   
   
# Performance

   * Modified Dietz Return
   * Capital gain/loss
   
# Documentation
The full documentation can be found in [Documentation](https://github.com/AmineAndam04/Algorithmic-trading/tree/master/Documentation)

 An exemple is  in [Exemple](https://github.com/AmineAndam04/Algorithmic-trading/tree/master/Exemple) (in frensh)
# How to use (Python 3)
First download the code of the library.it can be found in [AT](https://github.com/AmineAndam04/Algorithmic-trading/tree/master/AT) (or download the new version [AT](https://github.com/AmineAndam04/Algorithmic-trading/tree/master/AT_new_version)) and put it in your working directory or  use
```python
import os
path="C:/Users/pc/Desktop/..."  # the location of the downloaded code 
os.chdir(path)
```
Then import the libray : 
```python
import AT as at
```

# Genetic Algorithms
A genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection.(Mitchell, Melanie (1996). An Introduction to Genetic Algorithms)


In this project, we used the [DEAP library](https://deap.readthedocs.io/en/master/) to implement genetic algorithm.
In our case, we used those technics: 

                * Selection: Tournament selection (tools.selTournament)
                * Crossover: Simulated Binary Crossover (tools.cxSimulatedBinaryBounded)
                * Mutation: polynomial mutation (tools.mutPolynomialBounded )

# Updates 
Instead of creating just one module that contains all functions, we create a package [AT_new_version](https://github.com/AmineAndam04/Algorithmic-trading/tree/master/AT_new_version) that contain 5 modules. Each one has a specific task. 

There are 5 modules in the package: 

               * Indicators: to compute technical indicators
               * Signal: To generate trading signals 
               * GraphIndicators: To visualize technical indicators
               * GraphSignal: To visualize trading signals
               * Performance: Tools to evaluate trading strategies

# Sources
   * Eyal Wirsansky - Hands-On Genetic Algorithms with Python_ Applying genetic algorithms to solve real-world deep learning and artificial intelligence problems-Packt Publishing (2020)


   * Sebastien Donadio, Sourav Ghosh - Learn Algorithmic Trading_ Build and deploy algorithmic trading systems and strategies using Python and advanced data analysis-Packt Publishing (2019)


   * Prodromos E. Tsinaslanidis, Achilleas D. Zapranis - Technical Analysis for Algorithmic Pattern Recognition-Springer (2016)


# Credits
