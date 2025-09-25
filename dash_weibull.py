import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from scipy import stats
from scipy.special import gamma
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Distribucion de 2 y 3 parametros de Weibull", 
            style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
    
    html.Div([
        # Left panel for controls
        html.Div([
            html.H3("Parametros de la distribucion", style={'color': '#34495e', 'marginBottom': '20px'}),
            
            # Shape parameter
            html.Div([
                html.Label("Parametro de forma (k)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div([
                    dcc.Slider(
                        id='shape-slider',
                        min=0.1,
                        max=5.0,
                        step=0.1,
                        value=1.0,
                        marks={i: str(i) for i in [0.1, 1, 2, 3, 4, 5]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div([
                        html.Label("Valor:", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(
                            id='shape-input',
                            type='number',
                            value=1.0,
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            style={'width': '80px', 'height': '30px', 'fontSize': '12px'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
                ])
            ], style={'marginBottom': '20px'}),
            
            # Scale parameter
            html.Div([
                html.Label("Parametro de escala (λ)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div([
                    dcc.Slider(
                        id='scale-slider',
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        value=1.0,
                        marks={i: str(i) for i in [0.1, 1, 2, 5, 8, 10]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div([
                        html.Label("Value:", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(
                            id='scale-input',
                            type='number',
                            value=1.0,
                            min=0.1,
                            max=10.0,
                            step=0.1,
                            style={'width': '80px', 'height': '30px', 'fontSize': '12px'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
                ])
            ], style={'marginBottom': '20px'}),
            
            # Location parameter (for 3-parameter Weibull)
            html.Div([
                html.Label("Parametro de localizacion (γ)", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div([
                    dcc.Slider(
                        id='location-slider',
                        min=-2.0,
                        max=5.0,
                        step=0.1,
                        value=0.0,
                        marks={i: str(i) for i in [-2, -1, 0, 1, 2, 3, 4, 5]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div([
                        html.Label("Valor:", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(
                            id='location-input',
                            type='number',
                            value=0.0,
                            min=-2.0,
                            max=5.0,
                            step=0.1,
                            style={'width': '80px', 'height': '30px', 'fontSize': '12px'}
                        )
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
                ])
            ], style={'marginBottom': '20px'}),
            
            # Distribution type selector
            html.Div([
                html.Label("Tipo de distribucion", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.RadioItems(
                    id='distribution-type',
                    options=[
                        {'label': '2-Parametros', 'value': '2param'},
                        {'label': '3-Parametros', 'value': '3param'}
                    ],
                    value='2param',
                    style={'marginTop': '10px'}
                )
            ], style={'marginBottom': '20px'}),
            
            # Current parameter values display
            html.Div(id='parameter-display', style={
                'backgroundColor': '#f8f9fa',
                'padding': '15px',
                'borderRadius': '5px',
                'border': '1px solid #dee2e6'
            })
            
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 
                  'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px',
                  'marginRight': '20px'}),
        
        # Right panel for plots
        html.Div([
            # PDF plot
            html.H3("Funcion de densidad de probabilidad (PDF)", style={'color': '#34495e', 'marginBottom': '10px'}),
            dcc.Graph(id='pdf-plot', style={'height': '400px'}),
            
            # CDF plot
            html.H3("Funcion de distribucion acumulada (CDF)", style={'color': '#34495e', 'marginBottom': '10px', 'marginTop': '30px'}),
            dcc.Graph(id='cdf-plot', style={'height': '400px'})
            
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'maxHeight': '800px', 'overflow': 'auto'})
        
    ], style={'display': 'flex', 'marginBottom': '30px'}),
    
    # Statistics summary
    html.Div([
        html.H3("Estadisticas de la distribucion", style={'color': '#34495e', 'marginBottom': '15px'}),
        html.Div(id='stats-display', style={
            'backgroundColor': '#e8f4fd',
            'padding': '15px',
            'borderRadius': '5px',
            'border': '1px solid #bee5eb'
        })
    ]),
    
    # Footer
    html.Div([
        html.Hr(style={'border': '1px solid #dee2e6', 'margin': '30px 0 20px 0'}),
        html.P("© 2025 Jorge Eduardo Velasco Zavala", 
               style={'textAlign': 'center', 'color': '#6c757d', 'fontSize': '14px', 'margin': '5px 0'}),
        html.P("Para fines educativos exclusivamente. \n Este porgram es publico y ageno a cualquier partido político. \n Queda prohibido su uso para fines electorales", 
               style={'textAlign': 'center', 'color': '#6c757d', 'fontSize': '12px', 'margin': '5px 0'})
    ], style={'marginTop': '30px', 'paddingTop': '20px'})
    
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# Smart callback to handle bidirectional sync without circular dependencies
@callback(
    [Output('shape-slider', 'value'),
     Output('shape-input', 'value')],
    [Input('shape-slider', 'value'),
     Input('shape-input', 'value')],
    [State('shape-slider', 'value'),
     State('shape-input', 'value')]
)
def sync_shape_controls(slider_value, input_value, slider_state, input_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return 1.0, 1.0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'shape-slider':
        # Slider changed, update input
        validated_value = max(0.1, min(5.0, slider_value))
        return validated_value, validated_value
    else:
        # Input changed, update slider
        if input_value is None:
            return 1.0, 1.0
        validated_value = max(0.1, min(5.0, input_value))
        return validated_value, validated_value

@callback(
    [Output('scale-slider', 'value'),
     Output('scale-input', 'value')],
    [Input('scale-slider', 'value'),
     Input('scale-input', 'value')],
    [State('scale-slider', 'value'),
     State('scale-input', 'value')]
)
def sync_scale_controls(slider_value, input_value, slider_state, input_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return 1.0, 1.0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'scale-slider':
        # Slider changed, update input
        validated_value = max(0.1, min(10.0, slider_value))
        return validated_value, validated_value
    else:
        # Input changed, update slider
        if input_value is None:
            return 1.0, 1.0
        validated_value = max(0.1, min(10.0, input_value))
        return validated_value, validated_value

@callback(
    [Output('location-slider', 'value'),
     Output('location-input', 'value')],
    [Input('location-slider', 'value'),
     Input('location-input', 'value')],
    [State('location-slider', 'value'),
     State('location-input', 'value')]
)
def sync_location_controls(slider_value, input_value, slider_state, input_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return 0.0, 0.0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'location-slider':
        # Slider changed, update input
        validated_value = max(-2.0, min(5.0, slider_value))
        return validated_value, validated_value
    else:
        # Input changed, update slider
        if input_value is None:
            return 0.0, 0.0
        validated_value = max(-2.0, min(5.0, input_value))
        return validated_value, validated_value

# Callback to update parameter display
@callback(
    Output('parameter-display', 'children'),
    [Input('shape-slider', 'value'),
     Input('scale-slider', 'value'),
     Input('location-slider', 'value'),
     Input('distribution-type', 'value')]
)
def update_parameter_display(shape, scale, location, dist_type):
    if dist_type == '2param':
        return html.Div([
            html.P(f"Forma (k): {shape:.2f}", style={'margin': '5px 0'}),
            html.P(f"Escala (λ): {scale:.2f}", style={'margin': '5px 0'}),
            html.P("Localizacion (γ): 0.00 (fijo)", style={'margin': '5px 0', 'color': '#6c757d'})
        ])
    else:
        return html.Div([
            html.P(f"Forma (k): {shape:.2f}", style={'margin': '5px 0'}),
            html.P(f"Escala (λ): {scale:.2f}", style={'margin': '5px 0'}),
            html.P(f"Localizacion (γ): {location:.2f}", style={'margin': '5px 0'})
        ])

# Callback to update PDF plot
@callback(
    Output('pdf-plot', 'figure'),
    [Input('shape-slider', 'value'),
     Input('scale-slider', 'value'),
     Input('location-slider', 'value'),
     Input('distribution-type', 'value')]
)
def update_pdf_plot(shape, scale, location, dist_type):
    # Validate inputs to prevent division by zero
    shape = max(0.01, shape)  # Prevent shape parameter from being too small
    scale = max(0.01, scale)  # Prevent scale parameter from being too small
    
    # Generate x values
    if dist_type == '2param':
        x_min = max(0.01, location - 2)  # Ensure x > 0 for 2-parameter
        x_max = location + 8
    else:
        x_min = location - 2
        x_max = location + 8
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate PDF with error handling
    try:
        if dist_type == '2param':
            # 2-parameter Weibull: location = 0
            pdf = stats.weibull_min.pdf(x, shape, scale=scale)
        else:
            # 3-parameter Weibull: with location parameter
            pdf = stats.weibull_min.pdf(x - location, shape, scale=scale)
        
        # Handle any NaN or infinite values
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        
    except (ValueError, ZeroDivisionError):
        # Fallback to exponential distribution if Weibull fails
        pdf = np.zeros_like(x)
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        name='PDF',
        line=dict(color='#3498db', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f"PDF de Weibull (k={shape:.2f}, λ={scale:.2f}" + 
              (f", γ={location:.2f})" if dist_type == '3param' else ")"),
        xaxis_title="t",
        yaxis_title="Densidad de probabilidad",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                    range=[0, max(pdf) * 1.1] if max(pdf) > 0 else [0, 1])
    
    return fig

# Callback to update CDF plot
@callback(
    Output('cdf-plot', 'figure'),
    [Input('shape-slider', 'value'),
     Input('scale-slider', 'value'),
     Input('location-slider', 'value'),
     Input('distribution-type', 'value')]
)
def update_cdf_plot(shape, scale, location, dist_type):
    # Validate inputs to prevent division by zero
    shape = max(0.01, shape)  # Prevent shape parameter from being too small
    scale = max(0.01, scale)  # Prevent scale parameter from being too small
    
    # Generate x values
    if dist_type == '2param':
        x_min = max(0.01, location - 2)  # Ensure x > 0 for 2-parameter
        x_max = location + 8
    else:
        x_min = location - 2
        x_max = location + 8
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate CDF with error handling
    try:
        if dist_type == '2param':
            # 2-parameter Weibull: location = 0
            cdf = stats.weibull_min.cdf(x, shape, scale=scale)
        else:
            # 3-parameter Weibull: with location parameter
            cdf = stats.weibull_min.cdf(x - location, shape, scale=scale)
        
        # Handle any NaN or infinite values
        cdf = np.nan_to_num(cdf, nan=0.0, posinf=1.0, neginf=0.0)
        
    except (ValueError, ZeroDivisionError):
        # Fallback to uniform distribution if Weibull fails
        cdf = np.zeros_like(x)
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=cdf,
        mode='lines',
        name='CDF',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig.update_layout(
        title=f"CDF de Weibull (k={shape:.2f}, λ={scale:.2f}" + 
              (f", γ={location:.2f})" if dist_type == '3param' else ")"),
        xaxis_title="t",
        yaxis_title="Probabilidad acumulada",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                    range=[0, 1.05])
    
    return fig

# Callback to update statistics
@callback(
    Output('stats-display', 'children'),
    [Input('shape-slider', 'value'),
     Input('scale-slider', 'value'),
     Input('location-slider', 'value'),
     Input('distribution-type', 'value')]
)
def update_stats(shape, scale, location, dist_type):
    # Validate inputs to prevent division by zero
    shape = max(0.01, shape)  # Prevent shape parameter from being too small
    scale = max(0.01, scale)  # Prevent scale parameter from being too small
    
    try:
        if dist_type == '2param':
            # 2-parameter Weibull statistics
            mean = scale * gamma(1 + 1/shape)
            var = scale**2 * (gamma(1 + 2/shape) - (gamma(1 + 1/shape))**2)
            std = np.sqrt(max(0, var))  # Ensure variance is non-negative
            mode = scale * ((shape - 1) / shape)**(1/shape) if shape > 1 else 0
        else:
            # 3-parameter Weibull statistics
            mean = location + scale * gamma(1 + 1/shape)
            var = scale**2 * (gamma(1 + 2/shape) - (gamma(1 + 1/shape))**2)
            std = np.sqrt(max(0, var))  # Ensure variance is non-negative
            mode = location + scale * ((shape - 1) / shape)**(1/shape) if shape > 1 else location
        
        # Handle any NaN or infinite values
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
        mode = np.nan_to_num(mode, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        
    except (ValueError, ZeroDivisionError, OverflowError):
        # Fallback values if calculation fails
        mean = 0.0
        std = 0.0
        mode = 0.0
        var = 0.0
    
    return html.Div([
        html.Div([
            html.Div([
                html.H4("Media", style={'margin': '0', 'color': '#2c3e50'}),
                html.P(f"{mean:.4f}", style={'margin': '5px 0', 'fontSize': '18px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'margin': '5px'}),
            
            html.Div([
                html.H4("Desviacion estandar", style={'margin': '0', 'color': '#2c3e50'}),
                html.P(f"{std:.4f}", style={'margin': '5px 0', 'fontSize': '18px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'margin': '5px'}),
            
            html.Div([
                html.H4("Varianza", style={'margin': '0', 'color': '#2c3e50'}),
                html.P(f"{var:.4f}", style={'margin': '5px 0', 'fontSize': '18px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'margin': '5px'})
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
    ])

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)

server = app.server
