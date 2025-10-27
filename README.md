# Introduction to AI - Laboratory Work

This repository contains the laboratory assignments for the Introduction to Artificial Intelligence course. The labs cover fundamental AI concepts, algorithms, and techniques through practical implementations.

## 📚 Course Overview

This collection of labs provides hands-on experience with various AI topics including optimization methods, constraint satisfaction, evolutionary algorithms, machine learning, neural networks, reinforcement learning, and logic programming.

## 🗂️ Repository Structure

```
IntroToAI_Labs_PW/
├── Lab 1/ - Optimization Methods
├── Lab 2/ - Constraint Satisfaction Problems
├── Lab 3/ - Genetic Algorithms
├── Lab 4/ - Machine Learning & Classification
├── Lab 5/ - Neural Networks
├── Lab 6/ - Reinforcement Learning
└── Lab 7/ - Prolog Logic Programming
```

## 📝 Lab Descriptions

### Lab 1: Optimization Methods
**Topics:** Numerical optimization, Newton's method, gradient descent  
**Language:** Python  
**Key Concepts:**
- Implementation of Newton's method for optimization
- Gradient and Hessian computation using symbolic mathematics
- Finding extrema of multi-dimensional functions
- Convergence analysis and visualization

### Lab 2: Constraint Satisfaction Problems (CSP)
**Topics:** CSP solving, Sudoku solver, backtracking algorithms  
**Language:** Python  
**Key Concepts:**
- Constraint propagation and backtracking
- Domain reduction techniques
- Sudoku puzzle solving using CSP framework
- Validation and conflict detection

### Lab 3: Genetic Algorithms
**Topics:** Evolutionary computation, optimization  
**Language:** Python  
**Key Concepts:**
- Genetic algorithm implementation
- Roulette wheel selection
- Random interpolation crossover
- Gaussian mutation
- Optimization of the Booth function

### Lab 4: Machine Learning & Classification
**Topics:** Supervised learning, model selection, hyperparameter tuning  
**Language:** Python  
**Key Concepts:**
- Logistic Regression
- Random Forest Classifier
- Data preprocessing and feature engineering
- Cross-validation and model evaluation
- Working with real datasets (Titanic dataset)

### Lab 5: Neural Networks
**Topics:** Deep learning, multi-layer perceptrons  
**Language:** Python (PyTorch)  
**Key Concepts:**
- Building and training neural networks
- Hyperparameter tuning (learning rate, batch size, hidden layers)
- Different loss functions (Cross-Entropy, MSE, MAE)
- FashionMNIST dataset classification
- Model architecture experimentation

### Lab 6: Reinforcement Learning
**Topics:** Q-learning, value-based methods  
**Language:** Python (Gymnasium)  
**Key Concepts:**
- Q-learning algorithm implementation
- ε-greedy exploration strategy
- Epsilon decay scheduling
- Training agents in discrete environments (Taxi-v3)
- Performance evaluation and visualization

### Lab 7: Logic Programming
**Topics:** Declarative programming, natural language processing  
**Language:** Prolog  
**Key Concepts:**
- Prolog facts and rules
- Definite Clause Grammars (DCG)
- Natural language number parsing
- Converting English number words to numerical values
- Recursive pattern matching

## 🛠️ Prerequisites

### Python Labs (Labs 1-6)
- Python 3.7+
- Required packages (may vary by lab):
  - NumPy
  - Matplotlib
  - SymPy (Lab 1)
  - Pandas, Scikit-learn (Lab 4)
  - PyTorch, torchvision (Lab 5)
  - Gymnasium (Lab 6)

### Prolog Lab (Lab 7)
- SWI-Prolog or another Prolog interpreter

## 🚀 Getting Started

### Setting up Python Environment

1. Clone the repository:
```bash
git clone https://github.com/MimisGkolias/IntroToAI_Labs_PW.git
cd IntroToAI_Labs_PW
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages for specific labs:
```bash
# For Labs 1-3
pip install numpy matplotlib sympy

# For Lab 4
pip install numpy pandas scikit-learn matplotlib seaborn

# For Lab 5
pip install torch torchvision matplotlib numpy

# For Lab 6
pip install gymnasium numpy matplotlib
```

### Running the Labs

Navigate to the specific lab directory and run the Python files:
```bash
cd "Lab 1/lab1_cg105_g24_v4_Gkolias_Ntalas"
python lab1_cg105_g24_v4_Gkolias_Ntalas.py
```

For the Prolog lab:
```bash
cd "Lab 7/lab7_cg105_g24_v4_Gkolias_Ntalas"
swipl lab7_cg105_g24_v4_Gkolias_Ntalas.pl
```

## 📖 Additional Resources

Each lab directory contains:
- **Instruction PDFs**: Detailed assignment specifications and requirements
- **Report Instructions**: Guidelines for lab report submission
- **Implementation files**: Complete working solutions

## 👥 Authors

- **Group:** cg105_g24_v4
- **Members:** Gkolias, Ntalas

## 📄 License

This repository is for educational purposes as part of the Introduction to AI course.

## 🎓 Course Information

This work is part of the coursework for the Introduction to Artificial Intelligence course, covering essential topics in AI through practical programming assignments.
