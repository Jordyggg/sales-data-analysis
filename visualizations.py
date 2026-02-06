"""
Sales Data Visualizations Module
Custom visualization functions for sales analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class SalesVisualizer:
    """
    Visualization toolkit for sales data analysis
    """
    
    def __init__(self, data):
        """
        Initialize with sales data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Sales data
        """
        self.data = data
        self.fig_size = (12, 6)
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_revenue_trend(self, date_col='Date', revenue_col='Revenue', 
                          period='M', interactive=True):
        """
        Plot revenue trends over time
        
        Parameters:
        -----------
        date_col : str
            Date column name
        revenue_col : str
            Revenue column name
        period : str
            Aggregation period ('D', 'W', 'M', 'Q', 'Y')
        interactive : bool
            Use Plotly (True) or Matplotlib (False)
        """
        # Aggregate data
        df_agg = self.data.groupby(pd.Grouper(key=date_col, freq=period))[revenue_col].sum().reset_index()
        
        if interactive:
            fig = px.line(df_agg, x=date_col, y=revenue_col,
                         title=f'Revenue Trend ({period})',
                         labels={revenue_col: 'Revenue ($)', date_col: 'Date'},
                         template='plotly_white')
            
            fig.update_traces(line_color='#2E86AB', line_width=3)
            fig.update_layout(hovermode='x unified', height=500)
            fig.show()
        else:
            plt.figure(figsize=self.fig_size)
            plt.plot(df_agg[date_col], df_agg[revenue_col], 
                    color='#2E86AB', linewidth=2.5, marker='o')
            plt.title(f'Revenue Trend ({period})', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Revenue ($)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_category_distribution(self, category_col='Category', 
                                   revenue_col='Revenue', plot_type='pie'):
        """
        Plot revenue distribution by category
        
        Parameters:
        -----------
        category_col : str
            Category column name
        revenue_col : str
            Revenue column name
        plot_type : str
            'pie', 'bar', or 'donut'
        """
        category_revenue = self.data.groupby(category_col)[revenue_col].sum().sort_values(ascending=False)
        
        if plot_type == 'pie' or plot_type == 'donut':
            hole = 0.4 if plot_type == 'donut' else 0
            
            fig = px.pie(values=category_revenue.values, 
                        names=category_revenue.index,
                        title=f'Revenue Distribution by {category_col}',
                        hole=hole,
                        color_discrete_sequence=self.color_palette)
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.show()
            
        elif plot_type == 'bar':
            fig = px.bar(x=category_revenue.values, y=category_revenue.index,
                        orientation='h',
                        title=f'Revenue by {category_col}',
                        labels={'x': 'Revenue ($)', 'y': category_col},
                        color=category_revenue.values,
                        color_continuous_scale='Viridis')
            
            fig.update_layout(showlegend=False, height=500)
            fig.show()
    
    def plot_top_products(self, product_col='Product', revenue_col='Revenue', 
                         top_n=10):
        """
        Plot top N products by revenue
        """
        top_products = self.data.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(top_n)
        
        fig = px.bar(x=top_products.values, y=top_products.index,
                    orientation='h',
                    title=f'Top {top_n} Products by Revenue',
                    labels={'x': 'Revenue ($)', 'y': 'Product'},
                    color=top_products.values,
                    color_continuous_scale='Blues')
        
        fig.update_layout(showlegend=False, height=500)
        fig.show()
    
    def plot_regional_performance(self, region_col='Region', 
                                  revenue_col='Revenue', 
                                  units_col='Units_Sold'):
        """
        Plot regional performance with dual metrics
        """
        region_stats = self.data.groupby(region_col).agg({
            revenue_col: 'sum',
            units_col: 'sum'
        }).sort_values(revenue_col, ascending=False)
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Revenue by Region', 'Units Sold by Region'))
        
        fig.add_trace(
            go.Bar(x=region_stats.index, y=region_stats[revenue_col],
                  name='Revenue', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=region_stats.index, y=region_stats[units_col],
                  name='Units', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(showlegend=False, height=400, 
                         title_text='Regional Performance Analysis')
        fig.show()
    
    def plot_heatmap(self, index_col='Month_Name', columns_col='Category', 
                    values_col='Revenue'):
        """
        Create a heatmap for cross-tabulation analysis
        """
        pivot_table = self.data.pivot_table(
            values=values_col,
            index=index_col,
            columns=columns_col,
            aggfunc='sum',
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Revenue ($)'}, linewidths=0.5)
        plt.title(f'{values_col} Heatmap: {index_col} vs {columns_col}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(columns_col, fontsize=12)
        plt.ylabel(index_col, fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, numeric_cols=None):
        """
        Plot correlation matrix for numeric columns
        """
        if numeric_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        correlation = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.2f')
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_decomposition(self, date_col='Date', 
                                      value_col='Revenue', period=30):
        """
        Decompose time series into trend, seasonal, and residual
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare time series
        ts_data = self.data.groupby(date_col)[value_col].sum().sort_index()
        
        # Decompose
        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        decomposition.observed.plot(ax=axes[0], color='#2E86AB')
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
        
        decomposition.trend.plot(ax=axes[1], color='#A23B72')
        axes[1].set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=axes[2], color='#F18F01')
        axes[2].set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=axes[3], color='#6A994E')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.show()
    
    def plot_customer_segmentation(self, x_col='Customer_Age', 
                                   y_col='Revenue', 
                                   segment_col='Customer_Segment'):
        """
        Plot customer segmentation scatter plot
        """
        fig = px.scatter(self.data, x=x_col, y=y_col, 
                        color=segment_col,
                        size=y_col,
                        hover_data=[segment_col],
                        title='Customer Segmentation Analysis',
                        labels={x_col: x_col, y_col: 'Revenue ($)'},
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(height=600)
        fig.show()
    
    def plot_sales_funnel(self, stages, values):
        """
        Create a sales funnel visualization
        
        Parameters:
        -----------
        stages : list
            List of funnel stages
        values : list
            Values for each stage
        """
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": ["deepskyblue", "lightsalmon", "tan", "teal", "silver"]},
        ))
        
        fig.update_layout(title='Sales Funnel', height=500)
        fig.show()
    
    def plot_quarterly_comparison(self, year_col='Year', quarter_col='Quarter',
                                  revenue_col='Revenue'):
        """
        Compare quarterly performance across years
        """
        quarterly = self.data.groupby([year_col, quarter_col])[revenue_col].sum().reset_index()
        quarterly['Period'] = quarterly[year_col].astype(str) + '-Q' + quarterly[quarter_col].astype(str)
        
        fig = px.bar(quarterly, x='Period', y=revenue_col, color=year_col,
                    title='Quarterly Revenue Comparison',
                    labels={revenue_col: 'Revenue ($)'},
                    barmode='group',
                    color_discrete_sequence=['#A8DADC', '#457B9D'])
        
        fig.update_layout(height=450)
        fig.show()
    
    def create_dashboard(self):
        """
        Create a comprehensive dashboard with multiple plots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Category Distribution', 
                          'Regional Performance', 'Top Products'),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Revenue trend
        monthly_revenue = self.data.groupby(pd.Grouper(key='Date', freq='M'))['Revenue'].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=monthly_revenue['Date'], y=monthly_revenue['Revenue'],
                      mode='lines+markers', name='Revenue'),
            row=1, col=1
        )
        
        # Category pie
        category_revenue = self.data.groupby('Category')['Revenue'].sum()
        fig.add_trace(
            go.Pie(labels=category_revenue.index, values=category_revenue.values,
                  name='Category'),
            row=1, col=2
        )
        
        # Regional bar
        region_revenue = self.data.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=region_revenue.index, y=region_revenue.values, name='Region'),
            row=2, col=1
        )
        
        # Top products
        top_products = self.data.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(5)
        fig.add_trace(
            go.Bar(x=top_products.values, y=top_products.index, 
                  orientation='h', name='Products'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Sales Analytics Dashboard")
        fig.show()


def quick_visualize(data, plot_types=['trend', 'category', 'regional']):
    """
    Quick visualization with common plots
    
    Parameters:
    -----------
    data : pd.DataFrame
        Sales data
    plot_types : list
        Types of plots to generate
    """
    viz = SalesVisualizer(data)
    
    if 'trend' in plot_types:
        viz.plot_revenue_trend()
    
    if 'category' in plot_types:
        viz.plot_category_distribution()
    
    if 'regional' in plot_types:
        viz.plot_regional_performance()
    
    if 'products' in plot_types:
        viz.plot_top_products()
    
    if 'heatmap' in plot_types:
        viz.plot_heatmap()
    
    if 'correlation' in plot_types:
        viz.plot_correlation_matrix()


# Example usage
if __name__ == "__main__":
    print("Sales Data Visualizations Module")
    print("="*60)
    print("\nExample usage:")
    print("  from visualizations import SalesVisualizer")
    print("  viz = SalesVisualizer(df)")
    print("  viz.plot_revenue_trend()")
    print("  viz.plot_category_distribution()")
    print("\nOr use quick visualization:")
    print("  from visualizations import quick_visualize")
    print("  quick_visualize(df, ['trend', 'category', 'regional'])")
