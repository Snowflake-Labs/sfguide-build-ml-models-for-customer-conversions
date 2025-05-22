import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from snowflake.snowpark.context import get_active_session

# Set page configuration
st.set_page_config(
    page_title="Review Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title('📊 Review Analysis Dashboard')
st.markdown('Explore the relationships between review quality, sentiment, and purchase decisions.')

# Create bins for sentiment 
# Define sentiment order globally at the top of the script
sentiment_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

# Filter section
st.sidebar.header("Filters")

# Get Snowflake session
session = get_active_session()

# Load data from Snowflake
@st.cache_data(ttl=600)
def load_data():
    # Join the two tables on UUID
    query = """
    SELECT 
        t.UUID,
        t.PRODUCT_TYPE,
        t.PRODUCT_LAYOUT,
        t.PAGE_LOAD_TIME,
        t.PRODUCT_RATING,
        t.PURCHASE_DECISION,
        r.REVIEW_TEXT,
        r.REVIEW_QUALITY,
        r.REVIEW_SENTIMENT
    FROM 
        HOL_DB.HOL_SCHEMA.TABULAR_DATA t
    JOIN 
        HOL_DB.HOL_SCHEMA.REVIEWS r
    ON 
        t.UUID = r.UUID
    """
    df = session.sql(query).to_pandas()
    return df

# Load the data
with st.spinner("Loading data from Snowflake..."):
    df = load_data()

# Product Type filter
product_types = ["All"] + sorted(df["PRODUCT_TYPE"].unique().tolist())
selected_product_type = st.sidebar.selectbox("Product Type", product_types)

# Product Layout filter
layouts = ["All"] + sorted(df["PRODUCT_LAYOUT"].unique().tolist())
selected_layout = st.sidebar.selectbox("Product Layout", layouts)

# Review Quality filter
qualities = ["All"] + sorted(df["REVIEW_QUALITY"].unique().tolist())
selected_quality = st.sidebar.selectbox("Review Quality", qualities)

# Apply filters
filtered_df = df.copy()
if selected_product_type != "All":
    filtered_df = filtered_df[filtered_df["PRODUCT_TYPE"] == selected_product_type]
if selected_layout != "All":
    filtered_df = filtered_df[filtered_df["PRODUCT_LAYOUT"] == selected_layout]
if selected_quality != "All":
    filtered_df = filtered_df[filtered_df["REVIEW_QUALITY"] == selected_quality]

# Add debug info to help identify issues
st.sidebar.write("---")
st.sidebar.subheader("Data Statistics")
st.sidebar.write(f"Total reviews in current selection: {len(filtered_df)}")

# Count occurrences of each review quality
quality_counts = filtered_df["REVIEW_QUALITY"].value_counts().reset_index()
quality_counts.columns = ["Review Quality", "Count"]
st.sidebar.write("Review Quality Counts:")
st.sidebar.dataframe(quality_counts)

# Display data metrics in top row
st.header("Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    purchase_rate = filtered_df["PURCHASE_DECISION"].mean() * 100
    st.metric("Purchase Rate", f"{purchase_rate:.1f}%")

with col2:
    avg_sentiment = filtered_df["REVIEW_SENTIMENT"].mean()
    st.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")

with col3:
    avg_rating = filtered_df["PRODUCT_RATING"].mean()
    st.metric("Avg. Product Rating", f"{avg_rating:.1f}/5")

with col4:
    total_reviews = len(filtered_df)
    st.metric("Total Reviews", f"{total_reviews:,}")

# Main visualizations
st.header("Key Insights")

# Create two columns for the main charts
col1, col2 = st.columns(2)

# Chart 1: Review Quality vs Purchase Decision
with col1:
    st.subheader("Review Quality vs. Purchase Rate")
    
    # Group by Review Quality and calculate purchase rate
    quality_purchase = filtered_df.groupby("REVIEW_QUALITY").agg(
        mean=("PURCHASE_DECISION", "mean"),
        count=("PURCHASE_DECISION", "count")
    ).reset_index()

    # Ensure no missing data
    for quality in filtered_df["REVIEW_QUALITY"].unique():
        if quality not in quality_purchase["REVIEW_QUALITY"].values:
            # Add missing quality with 0 values
            quality_purchase = pd.concat([
                quality_purchase,
                pd.DataFrame({"REVIEW_QUALITY": [quality], "mean": [0], "count": [0]})
            ])

    quality_purchase["mean"] = quality_purchase["mean"] * 100  # Convert to percentage
    
    # Create bar chart
    chart1 = alt.Chart(quality_purchase).mark_bar().encode(
        x=alt.X('REVIEW_QUALITY:N', title='Review Quality'),
        y=alt.Y('mean:Q', title='Purchase Rate (%)'),
        color='REVIEW_QUALITY:N',
        tooltip=['REVIEW_QUALITY', alt.Tooltip('mean:Q', format='.1f', title='Purchase Rate (%)')]
    ).properties(
        title='How Review Quality Affects Purchase Rate',
        height=400
    )
    
    # Add text labels
    text1 = chart1.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text('mean:Q', format='.1f')
    )
    
    st.altair_chart(chart1 + text1, use_container_width=True)

# Chart 2: Sentiment vs Purchase Decision
with col2:
    st.subheader("Review Sentiment vs. Purchase Decision")
    
    # Create bins for sentiment
    filtered_df["Sentiment_Bin"] = pd.cut(
        filtered_df["REVIEW_SENTIMENT"],
        bins=5,
        labels=sentiment_order
    )
    
    # Ensure it's categorical with the right order
    filtered_df["Sentiment_Bin"] = pd.Categorical(
        filtered_df["Sentiment_Bin"],
        categories=sentiment_order,
        ordered=True
    )
    
    # Group by sentiment bins
    sentiment_purchase = filtered_df.groupby("Sentiment_Bin").agg(
        mean=("PURCHASE_DECISION", "mean"),
        count=("PURCHASE_DECISION", "count")
    ).reset_index()

    # Ensure all sentiment bins exist in the result
    for sentiment in sentiment_order:
        if sentiment not in sentiment_purchase["Sentiment_Bin"].values:
            # Add missing sentiment with 0 values
            sentiment_purchase = pd.concat([
                sentiment_purchase,
                pd.DataFrame({"Sentiment_Bin": [sentiment], "mean": [0], "count": [0]})
            ])

    # Convert to percentage
    sentiment_purchase["mean"] = sentiment_purchase["mean"] * 100

    # Ensure categorical order is maintained
    sentiment_purchase["Sentiment_Bin"] = pd.Categorical(
        sentiment_purchase["Sentiment_Bin"],
        categories=sentiment_order,
        ordered=True
    )

    # Sort by the categorical order
    sentiment_purchase = sentiment_purchase.sort_values("Sentiment_Bin")
    
    # Create bar chart
    # Define sentiment order
    sentiment_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    # Convert to categorical with specific order
    sentiment_purchase['Sentiment_Bin'] = pd.Categorical(
        sentiment_purchase['Sentiment_Bin'], 
        categories=sentiment_order, 
        ordered=True
    )
    
    chart2 = alt.Chart(sentiment_purchase).mark_bar().encode(
        x=alt.X('Sentiment_Bin:N', title='Review Sentiment', sort=None),  # Using sort=None to respect categorical order
        y=alt.Y('mean:Q', title='Purchase Rate (%)'),
        color=alt.Color('Sentiment_Bin:N', 
                       scale=alt.Scale(scheme='redblue'),
                       sort=None),  # Using sort=None here as well
        tooltip=['Sentiment_Bin', alt.Tooltip('mean:Q', format='.1f', title='Purchase Rate (%)')]
    ).properties(
        title='How Review Sentiment Affects Purchase Rate',
        height=400
    )
    
    # Add text labels
    text2 = chart2.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text('mean:Q', format='.1f')
    )
    
    st.altair_chart(chart2 + text2, use_container_width=True)

# Create a heatmap of Quality vs Sentiment and how it affects purchase decision
st.header("Combined Impact Analysis")

# Define the correct sentiment order globally
sentiment_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

# Prepare data for heatmap
# Convert the pivot table to a format suitable for Altair
heatmap_data = filtered_df.pivot_table(
    values='PURCHASE_DECISION',
    index='REVIEW_QUALITY', 
    columns='Sentiment_Bin',
    aggfunc='mean',
    fill_value=0  # Fill missing combinations with 0 instead of NaN
) * 100

# Make sure columns are in the right order
if isinstance(heatmap_data.columns, pd.CategoricalIndex):
    # Ensure all sentiment categories exist, add empty ones if missing
    for sentiment in sentiment_order:
        if sentiment not in heatmap_data.columns:
            heatmap_data[sentiment] = 0
    # Reorder columns
    heatmap_data = heatmap_data[sentiment_order]

# Reset index to convert to regular DataFrame
heatmap_df = heatmap_data.reset_index().melt(
    id_vars='REVIEW_QUALITY',
    var_name='Sentiment_Bin',
    value_name='Purchase_Rate'
)

# Fill any NaN values with 0
heatmap_df['Purchase_Rate'] = heatmap_df['Purchase_Rate'].fillna(0)

# Explicitly set the sentiment order using Categorical
heatmap_df['Sentiment_Bin'] = pd.Categorical(
    heatmap_df['Sentiment_Bin'],
    categories=sentiment_order,
    ordered=True
)

# Get a list of unique review qualities to maintain consistent order
quality_order = sorted(filtered_df['REVIEW_QUALITY'].unique())
heatmap_df['REVIEW_QUALITY'] = pd.Categorical(
    heatmap_df['REVIEW_QUALITY'],
    categories=quality_order,
    ordered=True
)

# Streamlit's native heatmap using Altair with explicit domain for sentiment
heatmap = alt.Chart(heatmap_df).mark_rect().encode(
    x=alt.X('Sentiment_Bin:N', 
            title='Review Sentiment',
            sort=None),  # Using 'sort=None' to keep the order defined by the Categorical
    y=alt.Y('REVIEW_QUALITY:N', 
            title='Review Quality',
            sort=None),  # Also preserving quality order
    color=alt.Color('Purchase_Rate:Q', 
                   scale=alt.Scale(scheme='redblue', reverse=True), 
                   title='Purchase Rate (%)'),
    tooltip=[
        alt.Tooltip('REVIEW_QUALITY:N', title='Review Quality'),
        alt.Tooltip('Sentiment_Bin:N', title='Review Sentiment'),
        alt.Tooltip('Purchase_Rate:Q', title='Purchase Rate (%)', format='.1f')
    ]
).properties(
    title="Combined Effect of Review Quality and Sentiment on Purchase Rate",
    height=400
)

# Add text overlay for the values - also with preserved ordering
text_overlay = alt.Chart(heatmap_df).mark_text(baseline='middle').encode(
    x=alt.X('Sentiment_Bin:N', sort=None),  # Keep consistent with heatmap
    y=alt.Y('REVIEW_QUALITY:N', sort=None),  # Keep consistent with heatmap
    text=alt.Text('Purchase_Rate:Q', format='.1f'),
    color=alt.condition(
        alt.datum.Purchase_Rate > 50,
        alt.value('white'),
        alt.value('black')
    )
)

st.altair_chart(heatmap + text_overlay, use_container_width=True)

# Additional insights section with expandable content
with st.expander("Detailed Analysis by Product Type"):
    # Create a chart showing purchase decision by product type and review quality
    product_quality_purchase = filtered_df.groupby(
        ["PRODUCT_TYPE", "REVIEW_QUALITY"]
    ).agg(
        Purchase_Rate=("PURCHASE_DECISION", "mean"),
        count=("PURCHASE_DECISION", "count")
    ).reset_index()
    
    # Convert to percentage
    product_quality_purchase["Purchase_Rate"] = product_quality_purchase["Purchase_Rate"] * 100
    
    # Get unique product types to maintain a consistent order
    product_types = sorted(filtered_df["PRODUCT_TYPE"].unique())
    product_quality_purchase["PRODUCT_TYPE"] = pd.Categorical(
        product_quality_purchase["PRODUCT_TYPE"],
        categories=product_types,
        ordered=True
    )
    
    chart4 = alt.Chart(product_quality_purchase).mark_bar().encode(
        x=alt.X('PRODUCT_TYPE:N', title='Product Type', sort=None),
        y=alt.Y('Purchase_Rate:Q', title='Purchase Rate (%)'),
        color=alt.Color('REVIEW_QUALITY:N', title='Review Quality'),
        tooltip=['PRODUCT_TYPE', 'REVIEW_QUALITY', alt.Tooltip('Purchase_Rate:Q', format='.1f')]
    ).properties(
        title='Purchase Rate by Product Type and Review Quality',
        height=400
    )
    
    st.altair_chart(chart4, use_container_width=True)
    
    # Create a scatter plot of sentiment vs purchase decision with product type
    chart5 = alt.Chart(filtered_df).mark_circle(opacity=0.7).encode(
        x=alt.X('REVIEW_SENTIMENT:Q', title='Review Sentiment'),
        y=alt.Y('PURCHASE_DECISION:Q', title='Purchase (1=Yes, 0=No)'),
        color='PRODUCT_TYPE:N',
        size=alt.Size('PRODUCT_RATING:Q', scale=alt.Scale(range=[50, 200])),
        tooltip=['PRODUCT_TYPE', 'PRODUCT_LAYOUT', 
                alt.Tooltip('REVIEW_SENTIMENT:Q', format='.2f'), 
                'PURCHASE_DECISION', 
                alt.Tooltip('PRODUCT_RATING:Q', format='.1f')]
    ).properties(
        title='Relationship Between Sentiment, Rating, and Purchase Decision',
        height=500
    )
    
    st.altair_chart(chart5, use_container_width=True)

# Text analysis feature (if there's review text)
if "REVIEW_TEXT" in filtered_df.columns:
    with st.expander("Review Text Analysis"):
        st.subheader("Sample Reviews by Sentiment")
        
        # Create sentiment categories
        sentiment_categories = {
            "Most Positive": filtered_df.nlargest(5, "REVIEW_SENTIMENT"),
            "Most Negative": filtered_df.nsmallest(5, "REVIEW_SENTIMENT"),
            "Positive with No Purchase": filtered_df[(filtered_df["REVIEW_SENTIMENT"] > 0.5) & (filtered_df["PURCHASE_DECISION"] == 0)].head(5),
            "Negative with Purchase": filtered_df[(filtered_df["REVIEW_SENTIMENT"] < 0) & (filtered_df["PURCHASE_DECISION"] == 1)].head(5)
        }
        
        # Create tabs for different sentiment categories
        tabs = st.tabs(list(sentiment_categories.keys()))
        
        for i, (category, data) in enumerate(sentiment_categories.items()):
            with tabs[i]:
                if not data.empty:
                    for _, row in data.iterrows():
                        st.markdown(f"""
                        **Sentiment Score:** {row['REVIEW_SENTIMENT']:.2f} | **Quality:** {row['REVIEW_QUALITY']} | **Purchased:** {'Yes' if row['PURCHASE_DECISION'] == 1 else 'No'}
                        
                        "{row['REVIEW_TEXT']}"
                        
                        ---
                        """)
                else:
                    st.write("No reviews in this category.")

# Footer
st.markdown("---")
st.caption("Dashboard built with Streamlit in Snowflake. Data from HOL_DB.HOL_SCHEMA tables.")