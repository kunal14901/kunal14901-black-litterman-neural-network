# Black-Litterman Model for Portfolio Optimization via Neural Network 🧠📈

An advanced portfolio optimization system that enhances the traditional Black-Litterman model with Q-Learning neural networks for dynamic view generation and confidence estimation on NIFTY 50 stocks.

## 🎯 Project Overview

This master's thesis project revolutionizes portfolio optimization by replacing manual subjective views in the Black-Litterman model with an intelligent reinforcement learning agent. The system learns from historical price patterns and market signals to dynamically generate views and confidence scores, enabling adaptive asset allocation.

**Academic Context:** IIT Kharagpur Master's Thesis Project  

## 🚀 Key Innovation

Traditional Black-Litterman models rely on manual view assignment, which is subjective and static. This project introduces:
- **Automated View Generation** using Q-Learning neural networks
- **Dynamic Confidence Scoring** based on market patterns
- **Adaptive Integration** of market behavior into asset allocation

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | 1.10 |
| **Annual Return** | 21.25% |
| **Annualized Volatility** | 17.6% |
| **Max Drawdown** | 18.3% |
| **Benchmark** | NIFTY 50 |

## 🔧 Technical Architecture

### Core Components

1. **Q-Learning Neural Network**
   - Learns optimal view generation from historical data
   - Dynamically estimates confidence levels
   - Adapts to changing market conditions

2. **Enhanced Black-Litterman Framework**
   - Incorporates market-implied risk aversion
   - Derives equilibrium returns from market data
   - Solves for posterior returns using Bayesian updating

3. **Robust Covariance Estimation**
   - Ledoit-Wolf shrinkage for stable covariance matrices
   - Reduces estimation error in high-dimensional settings

4. **Backtesting Engine**
   - Backtrader integration for historical simulation
   - Performance benchmarking against NIFTY 50

### Mathematical Framework

```
π* = τΣ(τΣ + Ω)^(-1)μ + (τΣ + Ω)^(-1)ΩΠ
```

Where:
- π*: Posterior expected returns
- τ: Scaling factor for uncertainty
- Σ: Covariance matrix (Ledoit-Wolf shrinkage)
- Ω: Confidence matrix (Neural Network generated)
- μ: Market-implied equilibrium returns
- Π: View portfolio (Q-Learning generated)

## 🛠️ Technologies Used

- **Python**: Core implementation language
- **TensorFlow/PyTorch**: Neural network framework
- **NumPy/SciPy**: Numerical computations
- **Pandas**: Data manipulation
- **Backtrader**: Backtesting framework
- **Scikit-learn**: ML utilities
- **Matplotlib/Seaborn**: Visualization

## 📈 Key Features

### 1. Intelligent View Generation
- **Q-Learning Agent** learns optimal views from market data
- **Pattern Recognition** identifies profitable market signals
- **Adaptive Learning** adjusts to market regime changes

### 2. Dynamic Confidence Estimation
- **Neural Network** quantifies view confidence
- **Historical Validation** based on past performance
- **Risk-Adjusted Scoring** considers market volatility

### 3. Robust Portfolio Construction
- **Ledoit-Wolf Shrinkage** for stable covariance estimation
- **Market-Implied Risk Aversion** from equilibrium theory
- **Bayesian Updating** for posterior return estimation

### 4. Comprehensive Backtesting
- **Historical Simulation** on NIFTY 50 constituents
- **Performance Analytics** with risk-adjusted metrics
- **Benchmark Comparison** against market index

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scipy scikit-learn tensorflow backtrader matplotlib seaborn
```

### Quick Start
```python
from black_litterman_nn import BlackLittermanNN
from data_loader import load_nifty50_data

# Load NIFTY 50 data
data = load_nifty50_data('2018-01-01', '2024-12-31')

# Initialize model
bl_nn = BlackLittermanNN(
    lookback_window=252,  # 1 year
    neural_network_config={'layers': [64, 32], 'activation': 'relu'},
    q_learning_params={'learning_rate': 0.001, 'epsilon': 0.1}
)

# Train the model
bl_nn.train(data)

# Generate portfolio allocation
weights = bl_nn.optimize_portfolio(current_data)
```

## 📊 Research Methodology

### 1. Data Preparation
- Historical price data for NIFTY 50 constituents
- Market capitalization weights for equilibrium returns
- Technical indicators for feature engineering

### 2. Model Training
- **Q-Learning Training** on historical price patterns
- **Neural Network Optimization** for confidence estimation
- **Cross-validation** for hyperparameter tuning

### 3. Portfolio Optimization
- **View Generation** using trained Q-Learning agent
- **Confidence Scoring** via neural network
- **Portfolio Construction** using Black-Litterman framework

### 4. Performance Evaluation
- **Backtesting** on out-of-sample data
- **Risk-adjusted Performance** metrics
- **Statistical Significance** testing

## 🔬 Research Contributions

1. **Novel Integration** of reinforcement learning with classical portfolio theory
2. **Automated View Generation** eliminating subjective bias
3. **Dynamic Adaptation** to changing market conditions
4. **Empirical Validation** on Indian equity markets

## 📚 Skills Demonstrated

- **Quantitative Research** & Financial Modeling
- **Machine Learning** & Deep Learning
- **Portfolio Optimization** & Risk Management
- **Statistical Analysis** & Numerical Methods
- **Python Programming** & Software Development

## 🎓 Academic Impact

This research bridges the gap between traditional finance theory and modern machine learning, demonstrating how artificial intelligence can enhance classical portfolio optimization techniques for better risk-adjusted returns.

## 📄 Citation

If you use this work in your research, please cite:
```
@mastersthesis{blacklitterman2025,
  title={Black-Litterman Model for Portfolio Optimization via Neural Network},
  author={[Kunal Kumar]},
  school={Indian Institute of Technology, Kharagpur},
  year={2025}
}
```

## 🤝 Contributing

This is an academic research project. For collaboration or questions, please contact through appropriate academic channels.

⭐ **Star this repository** if you find it useful for your research!

*This project demonstrates the successful integration of modern machine learning techniques with classical financial theory, achieving superior risk-adjusted returns through intelligent automation.*
