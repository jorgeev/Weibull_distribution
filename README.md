# Weibull Distribution Explorer

A comprehensive interactive dashboard built with Dash and Plotly for exploring the Weibull distribution and understanding how its shape varies with different parameters.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Interactive-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Dash](https://img.shields.io/badge/Dash-2.14+-red) ![Plotly](https://img.shields.io/badge/Plotly-5.17+-purple)

## 🎯 Overview

The Weibull Distribution Explorer is an educational tool designed to help users understand how the Weibull distribution changes as you modify its key parameters. The dashboard provides real-time visualization of both 2-parameter and 3-parameter Weibull distributions with interactive controls and statistical summaries.

## ✨ Features

### 🔧 Interactive Controls
- **Shape Parameter (β)**: Adjust from 0.1 to 5.0 to control distribution shape
  - Use slider for visual adjustment or text input for precise values
- **Scale Parameter (λ)**: Modify from 0.1 to 10.0 to control distribution spread
  - Dual input method: slider + direct numerical input
- **Location Parameter (γ)**: Shift distribution from -2.0 to 5.0 (3-parameter mode)
  - Synchronized slider and text input controls
- **Distribution Type Toggle**: Switch between 2-parameter and 3-parameter Weibull distributions
- **Bidirectional Input**: Sliders and text boxes stay synchronized - change one, the other updates automatically

### 📊 Real-time Visualizations
- **Probability Density Function (PDF)**: Live updates showing distribution shape
- **Cumulative Distribution Function (CDF)**: Interactive cumulative probability curves
- **Auto-scaling**: Intelligent y-axis scaling to prevent visualization issues

### 📈 Statistical Analysis
- **Real-time Statistics**: Mean, Standard Deviation, Mode, and Variance
- **Parameter Display**: Current parameter values with clear labeling
- **Mathematical Accuracy**: Proper gamma function calculations for precise statistics

### 🎨 Modern UI Design
- **Responsive Layout**: Clean, professional interface with intuitive controls
- **Color-coded Sections**: Organized layout with visual hierarchy
- **Fixed Plot Heights**: Consistent visualization without vertical overflow
- **Professional Footer**: Copyright and educational purpose statements

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd weibull-distribution-explorer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   python dash_weibull.py
   ```

4. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8050`
   - The dashboard will be ready for interactive exploration!

## 📚 Understanding Weibull Distributions

### 2-Parameter Weibull Distribution
- **Shape Parameter (β)**: Controls the hazard rate
  - β < 1: Decreasing hazard rate (early failures)
  - β = 1: Constant hazard rate (exponential distribution)
  - β > 1: Increasing hazard rate (wear-out failures)
- **Scale Parameter (λ)**: Controls the spread and scale of the distribution

### 3-Parameter Weibull Distribution
- **Additional Location Parameter (γ)**: Shifts the distribution along the x-axis
- **Applications**: Useful for modeling systems with minimum lifetime or threshold effects

### Mathematical Formulation
The Weibull PDF is given by:
```
f(x) = (β/λ) * ((x-γ)/λ)^(β-1) * exp(-((x-γ)/λ)^β)
```

## 🛠️ Technical Details

### Dependencies
- **Dash 2.14.2**: Web application framework
- **Plotly 5.17.0**: Interactive plotting library
- **NumPy 1.24.3**: Numerical computing
- **SciPy 1.11.1**: Scientific computing (including gamma function)
- **Pandas 2.0.3**: Data manipulation

### Architecture
- **Frontend**: Dash components with Plotly visualizations
- **Backend**: Python callbacks for real-time updates
- **Statistics**: SciPy for mathematical calculations
- **Layout**: Responsive design with fixed constraints

### Key Functions
- `update_pdf_plot()`: Generates PDF visualizations
- `update_cdf_plot()`: Creates CDF plots
- `update_stats()`: Calculates distribution statistics
- `update_parameter_display()`: Shows current parameter values
- **Bidirectional Sync Functions**: 
  - `update_shape_input()` / `update_shape_slider()`: Sync shape parameter controls
  - `update_scale_input()` / `update_scale_slider()`: Sync scale parameter controls
  - `update_location_input()` / `update_location_slider()`: Sync location parameter controls

## 🎓 Educational Applications

### Learning Objectives
- Understand how parameters affect distribution shape
- Visualize the relationship between PDF and CDF
- Explore statistical properties of Weibull distributions
- Compare 2-parameter vs 3-parameter models
- Practice precise parameter input and validation
- Learn about bidirectional control synchronization

### Use Cases
- **Reliability Engineering**: Failure time analysis
- **Quality Control**: Process capability studies
- **Survival Analysis**: Time-to-event modeling
- **Statistical Education**: Distribution theory and applications

## 🔧 Customization

### Modifying Parameter Ranges
Edit the slider and input configurations in the layout section:
```python
# Slider configuration
dcc.Slider(
    id='shape-slider',
    min=0.1,        # Minimum value
    max=5.0,        # Maximum value
    step=0.1,       # Step size
    value=1.0,      # Default value
)

# Corresponding text input
dcc.Input(
    id='shape-input',
    type='number',
    value=1.0,
    min=0.1,
    max=5.0,
    step=0.1,
    style={'width': '80px', 'height': '30px', 'fontSize': '12px'}
)
```

### Adding New Input Controls
To add new parameter controls with bidirectional sync:
1. Add slider and input components to the layout
2. Create corresponding callback functions for bidirectional sync
3. Update the validation ranges in the callback functions

### Adding New Statistics
Extend the `update_stats()` function to include additional statistical measures:
```python
# Add new statistics to the return statement
html.P(f"Skewness: {skewness:.4f}", style={...})
```

### Styling Modifications
Update the CSS styles in the layout components to match your preferred design:
```python
style={'backgroundColor': '#your-color', 'padding': '20px'}
```

## 📝 License

This project is for educational purposes only. The code is provided as-is for learning and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📞 Support

For questions or support regarding this educational tool, please refer to the documentation or create an issue in the repository.

---

**© 2025 Weibull Distribution Explorer - For Educational Purposes Only**
