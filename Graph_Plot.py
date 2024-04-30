import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Read the data from the CSV file with specified encoding
data = pd.read_csv("./Data/graph.csv", encoding='latin1')

# Extract the columns for predicted and actual values
predicted_values = data["Predicted Annual Extractable Ground Water Resource"]
actual_values = data["Annual Extractable Ground Water Resource"]

# Create the plot function
@st.cache_data(show_spinner=False)
def plot_data(x_min, x_max):
    fig, ax = plt.subplots(figsize=(16, 8))  # Larger figure size
    ax.plot(predicted_values, label='Predicted', marker='o')
    ax.plot(actual_values, label='Actual', marker='s')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Annual Extractable Ground Water Resource')
    ax.set_title('Actual vs Predicted Annual Extractable Ground Water Resource')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(x_min, x_max + 1))
    ax.set_xlim(x_min, x_max)
    
    # Convert the plot to a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return buf

# Create the Streamlit app
def main():
    st.title('Graph Plot')

    st.write("Plot of Actual Annual Extractable Ground Water Resource vs Predicted Annual Extractable Ground Water Resource:")
    
    # Add a slider for controlling zoom level
   
    x_min, x_max = st.slider("X-axis range", 0, len(predicted_values), (0, len(predicted_values)), 1)
    
    # Display the plot with the selected zoom level
    fig = plot_data(x_min, x_max)
    st.image(fig, use_column_width=True)

if __name__ == "__main__":
    main()
