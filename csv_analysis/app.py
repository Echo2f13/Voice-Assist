import streamlit as st
from os import makedirs
from os.path import join, exists
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import inspect

# Set up paths
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, join(parent_dir, 'preprocess'))
from data.preprocess import CSVPreProcess

# Constants
VIZ_ROOT = 'Plots'
NUNIQUE_THRESHOLD = 20

def visualize_csv(csv, target_col=None, index_column=None, exclude_columns=[], save=False, show=True):
    """Generates various visualization charts and displays them in Streamlit."""
    visualizer = CSVVisualize(csv, target_col=target_col, 
                            index_column=index_column, 
                            exclude_columns=exclude_columns)
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Correlation", "📈 Scatter Plots", "📉 Distributions", 
        "📦 Box/Violin Plots", "🥧 Composition", "📐 Regression", 
        "🌊 KDE Plots", "📌 Stem Plots", "📈 Line Charts"
    ])
    
    with tab1:
        st.header("Correlation Analysis")
        visualizer.plot_correlation_map(save=save, show=show)
        visualizer.plot_diagonal_correlation_matrix(save=save, show=show)
    
    with tab2:
        st.header("Scatter Plots")
        visualizer.plot_scatter_plot_matrix(save=save, show=show)
        visualizer.plot_scatter_plots(save=save, show=show)
        visualizer.plot_scatter_plot_with_categorical(save=save, show=show)
    
    with tab3:
        st.header("Distribution Analysis")
        visualizer.plot_histogram(save=save, show=show)
        visualizer.plot_kde(save=save, show=show)
    
    with tab4:
        st.header("Box & Violin Plots")
        visualizer.plot_horizontal_box_plot(save=save, show=show)
    
    with tab5:
        st.header("Composition Charts")
        visualizer.plot_pie_chart(save=save, show=show)
    
    with tab6:
        st.header("Regression Analysis")
        visualizer.plot_regression_marginals(save=save, show=show)
    
    with tab7:
        st.header("Kernel Density Estimates")
        visualizer.plot_kde(save=save, show=show)
    
    with tab8:
        st.header("Stem Plots")
        visualizer.plot_stem_plots(save=save, show=show)
    

class CSVVisualize:
    def __init__(self, input, target_col=None, index_column=None, exclude_columns=[]):
        """Initialize the visualizer with data and configuration."""
        if isinstance(input, str):
            self.df = pd.read_csv(input, index_col=index_column)
        else:
            self.df = input
        
        self.df.drop(exclude_columns, inplace=True, errors='ignore')
        self.col_names = list(self.df.columns)
        self.target_column = self.col_names[-1] if target_col is None else target_col
        self.df.dropna(subset=[self.target_column], inplace=True)
        self.num_cols = len(self.col_names)
        self.output_format = 'png'
        self.categorical_data_types = ['object', 'str']
        
        viz = CSVPreProcess(self.df, target_col=target_col, index_column=index_column)
        self.df = viz.fill_numerical_na(ret=True)
        self.df = viz.fill_categorical_na(ret=True)
        
        self.categorical_column_list = []
        self.populate_categorical_column_list()
        self.numerical_column_list = list(self.get_filtered_dataframe(include_type=np.number))
        temp_col_list = [num_col for num_col in self.numerical_column_list 
                        if self.df[num_col].nunique() < NUNIQUE_THRESHOLD]
        self.continuous_column_list = [x for x in self.numerical_column_list if x not in temp_col_list]
        self.non_continuous_col_list = self.categorical_column_list + temp_col_list

    def save_or_show(self, plot, plot_type, file_name, x_label=None, y_label=None, save=False, show=True):
        """Display or save the visualization."""
        if show:
            fig = None
            
            if hasattr(plot, 'figure'):  # Seaborn plots
                fig = plot.figure
            elif isinstance(plot, plt.Figure):  # Matplotlib figure
                fig = plot
            elif hasattr(plot, 'fig'):  # Some objects have fig attribute
                fig = plot.fig
            
            if fig is not None:
                if x_label:
                    fig.axes[0].set_xlabel(x_label)
                if y_label:
                    fig.axes[0].set_ylabel(y_label)
                fig.suptitle(f"{plot_type}: {file_name}")
                st.pyplot(fig)
            else:
                if x_label:
                    plt.xlabel(x_label)
                if y_label:
                    plt.ylabel(y_label)
                plt.title(f"{plot_type}: {file_name}")
                st.pyplot()
        
        if save:
            save_dir = join(VIZ_ROOT, plot_type)
            if not exists(save_dir):
                makedirs(save_dir)
            save_path = join(save_dir, file_name)
            
            if hasattr(plot, 'savefig'):
                plot.savefig(save_path)
            else:
                plt.savefig(save_path)
        
        plt.clf()

    def get_filtered_dataframe(self, include_type=[], exclude_type=[]):
        """Filter DataFrame by column dtypes."""
        if include_type or exclude_type:
            return self.df.select_dtypes(include=include_type, exclude=exclude_type)
        return self.df

    def populate_categorical_column_list(self):
        """Identify categorical columns."""
        df = self.get_filtered_dataframe(exclude_type=np.number)
        if not self.categorical_column_list:
            for column in df:
                if df[column].nunique() <= NUNIQUE_THRESHOLD:
                    self.categorical_column_list.append(column)

    def get_categorical_numerical_columns_pairs(self):
        """Generate pairs of categorical and numerical columns."""
        paired_column_list = list(itertools.product(self.categorical_column_list, self.col_names))
        return [(cat, num) for cat, num in paired_column_list if cat != num]

    def get_correlated_numerical_columns(self, min_absolute_coeff=0.3):
        """Find correlated numerical columns."""
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        new_columns = list(df_new.columns)
        result_paired_columns = []
        
        for col1, col2 in itertools.product(new_columns, new_columns):
            if col1 != col2:
                try:
                    if abs(df_new[col1].corr(df_new[col2])) >= float(min_absolute_coeff):
                        result_paired_columns.append((col1, col2))
                except Exception as e:
                    st.warning(f'Error checking correlation for {col1}, {col2}: {e}')
        return result_paired_columns

    def plot_correlation_map(self, save=False, show=True):
        """Plot correlation heatmaps."""
        df_num = self.get_filtered_dataframe(include_type=np.number)
        df_cont = self.df[self.continuous_column_list]
        
        st.subheader("All Numerical Columns")
        corr_matrix_num = df_num.corr()
        plot = sns.heatmap(corr_matrix_num, annot=True, fmt=".2f", cmap='coolwarm')
        self.save_or_show(plot, 'correlation_map', 'all_numerical_cols', save=save, show=show)
        
        st.subheader("Continuous Columns Only")
        corr_matrix_cont = df_cont.corr()
        plot = sns.heatmap(corr_matrix_cont, annot=True, fmt=".2f", cmap='coolwarm')
        self.save_or_show(plot, 'correlation_map', 'continuous_cols', save=save, show=show)

    def plot_scatter_plots(self, save=False, show=True):
        """Generate scatter plots."""
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())
        
        for i, (x, y) in enumerate(col_pairs):
            try:
                st.caption(f"{x} vs {y}")
                sns_plot = sns.scatterplot(x=x, y=y, data=self.df, alpha=0.6)
                self.save_or_show(sns_plot, 'scatter', f'{x}_{y}', x, y, save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot scatter plot for {x}, {y}: {e}')

    def plot_horizontal_box_plot(self, save=False, show=True):
        """Generate box and violin plots."""
        new_df = self.df.copy()
        cat_cols = self.non_continuous_col_list
        num_cols = self.numerical_column_list
        cont_cols = self.continuous_column_list
        
        st.subheader("Single Variable Plots")
        for x_col in cont_cols:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                sns.boxplot(x=x_col, data=self.df, ax=ax1)
                sns.violinplot(x=x_col, data=self.df, ax=ax2)
                self.save_or_show(fig, 'box_violin_plot', x_col, save=save, show=show)
            except Exception as e:
                st.warning(f'Error plotting for {x_col}: {e}')
        
        st.subheader("Grouped Plots")
        for y_col in cat_cols:
            for x_col in cont_cols:
                try:
                    if y_col in num_cols:
                        new_df[y_col] = new_df[y_col].astype('category')
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    sns.boxplot(x=x_col, y=y_col, data=new_df, ax=ax1)
                    sns.violinplot(x=x_col, y=y_col, data=new_df, ax=ax2)
                    self.save_or_show(fig, 'box_violin_plot', f'{x_col}_{y_col}', save=save, show=show)
                except Exception as e:
                    st.warning(f'Error plotting for {x_col}, {y_col}: {e}')

    def plot_regression_marginals(self, save=False, show=True):
        """Plot regression with marginal distributions."""
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())
        
        for x, y in col_pairs:
            try:
                st.caption(f"{x} vs {y}")
                sns_plot = sns.jointplot(x=x, y=y, data=df_new, kind="reg", 
                                        truncate=False, height=6, ratio=4)
                self.save_or_show(sns_plot, 'regression_marginals', f'{x}_{y}', 
                                x, y, save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot regression for {x}, {y}: {e}')

    def plot_scatter_plot_with_categorical(self, save=False, show=True):
        """Plot categorical scatter plots with improved handling."""
        cat_cols = self.non_continuous_col_list
        num_cols = self.continuous_column_list
        
        for cat_col in cat_cols:
            for num_col in num_cols:
                try:
                    st.caption(f"{cat_col} vs {num_col}")
                    
                    # Use stripplot for large datasets, swarmplot for small ones
                    if len(self.df) > 100:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.stripplot(x=cat_col, y=num_col, data=self.df, 
                                    ax=ax, jitter=0.3, size=3, alpha=0.6)
                        ax.set_title("Strip Plot (large dataset)")
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.swarmplot(x=cat_col, y=num_col, data=self.df, 
                                    ax=ax, size=3)
                        ax.set_title("Swarm Plot")
                    
                    self.save_or_show(fig, 'categorical_scatter', 
                                    f'{cat_col}_{num_col}', save=save, show=show)
                except Exception as e:
                    st.warning(f'Error plotting categorical scatter for {cat_col}, {num_col}: {e}')

    def plot_scatter_plot_matrix(self, hue_col_list=[], save=False, show=True):
        """Plot scatter plot matrix."""
        for col in self.categorical_column_list[:3]:  # Limit to 3 hue variables
            try:
                st.subheader(f"Scatter Matrix (hue: {col})")
                sns_plot = sns.pairplot(self.df, 
                                       vars=self.continuous_column_list[:5],  # Limit columns
                                       hue=col, 
                                       plot_kws={'alpha':0.6, 's':20},
                                       height=2)
                self.save_or_show(sns_plot, 'scatterplot_matrix', f'hue_{col}', 
                                save=save, show=show)
            except Exception as e:
                st.warning(f'Error plotting scatter matrix with hue {col}: {e}')

    def plot_pie_chart(self, x=None, y=None, save=False, show=True, threshold=10):
        """Plot pie charts for categorical data."""
        df_new = self.df[self.non_continuous_col_list]
        
        for col in df_new.columns:
            try:
                val_counts = df_new[col].value_counts()
                if len(val_counts) > 10:  # Skip if too many categories
                    st.warning(f"Skipping pie chart for {col} - too many categories ({len(val_counts)})")
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 5))
                val_counts.plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
                ax.set_title(f"Distribution of {col}")
                self.save_or_show(fig, 'piechart', col, save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot pie chart for {col}: {e}')

    def plot_histogram(self, save=False, show=True):
        """Plot histograms for numerical data."""
        df = self.get_filtered_dataframe(include_type=np.number)
        
        for column in df.columns[:10]:  # Limit to first 10 columns
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df[column], kde=True, ax=ax)
                ax.set_title(f"Distribution of {column}")
                self.save_or_show(fig, 'histogram', column, save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot histogram for {column}: {e}')

    def plot_line_chart(self, save=False, show=True):
        """Plot line charts for sequential data."""
        xs = []
        for col in self.col_names:
            if self.df[col].shape[0] == self.df[col].unique().shape[0]:
                xs.append(col)
        
        for x in xs[:3]:  # Limit to 3 line charts
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.lineplot(x=x, y=self.target_column, data=self.df, ax=ax)
                ax.set_title(f"{self.target_column} over {x}")
                self.save_or_show(fig, 'line_chart', f'{x}_vs_{self.target_column}', 
                                save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot line chart for {x}: {e}')

    def plot_diagonal_correlation_matrix(self, save=False, show=True):
        """Plot triangular correlation matrix."""
        try:
            corr = self.df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                       square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            ax.set_title("Diagonal Correlation Matrix")
            self.save_or_show(fig, 'Diagonal_correlation_matrix', 
                            'Diagonal_correlation_matrix', save=save, show=show)
        except Exception as e:
            st.warning(f'Cannot plot diagonal correlation matrix: {e}')

    def plot_stem_plots(self, save=False, show=True):
        """Plot stem plots."""
        df_new = self.get_filtered_dataframe(include_type=[np.number])
        col_pairs = self.get_correlated_numerical_columns(min_absolute_coeff=0.5)
        col_pairs.extend(self.get_categorical_numerical_columns_pairs())
        
        for x, y in col_pairs[:5]:  # Limit to 5 stem plots
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.stem(df_new[x], df_new[y], linefmt='grey', markerfmt='o', basefmt=' ')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"Stem Plot: {x} vs {y}")
                self.save_or_show(fig, 'stem', f'{x}_{y}', save=save, show=show)
            except Exception as e:
                st.warning(f'Cannot plot stem plot for {x}, {y}: {e}')

    def plot_kde(self, save=False, show=True):
        """Plot kernel density estimates."""
        col_names = self.numerical_column_list
        
        for i in range(min(3, len(col_names))):  # Limit to 3 KDE plots
            for j in range(i + 1, min(i + 4, len(col_names))):  # Limit combinations
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.kdeplot(data=self.df, x=col_names[i], y=col_names[j], 
                               ax=ax, cmap="Blues", shade=True)
                    ax.set_title(f"KDE: {col_names[i]} vs {col_names[j]}")
                    self.save_or_show(fig, 'KDE Chart', f'{col_names[i]}_{col_names[j]}', 
                                    save=save, show=show)
                except Exception as e:
                    st.warning(f'Cannot plot KDE for {col_names[i]}, {col_names[j]}: {e}')

def main():
    st.set_page_config(page_title="Data Visualizer", layout="wide")
    st.title("📊 Interactive Data Visualizer")
    
    # File upload and configuration - moved to main page
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully!")
            
            # Configuration options in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_col = st.selectbox("Select target column", 
                                        df.columns, 
                                        index=len(df.columns)-1)
            
            with col2:
                exclude_cols = st.multiselect("Select columns to exclude", 
                                             df.columns)
            
            with col3:
                index_col = st.selectbox("Select index column (optional)", 
                                        [None] + list(df.columns))
            
            if st.button("✨ Generate Visualizations", type="primary"):
                with st.spinner("Creating visualizations..."):
                    visualize_csv(df, 
                                 target_col=target_col, 
                                 exclude_columns=exclude_cols,
                                 index_column=index_col)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    else:
        st.info("ℹ️ Please upload a CSV file to get started")

if __name__ == "__main__":
    main()