import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🍺 Beer Servings & Alcohol Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px;'>Predict total litres of pure alcohol based on beverage servings.</p>", unsafe_allow_html=True)

# Load dataset & model
df = pd.read_csv("cleaned_drinks.csv")
model = pickle.load(open("best_model.pkl", "rb"))

# Full-width 2-column layout
input_col, output_col = st.columns([2, 3])

# LEFT COLUMN: Inputs & Prediction Result
with input_col:
    st.header("Input Details")

    beer_servings = st.number_input("Beer Servings", min_value=0, max_value=400, value=100)
    spirit_servings = st.number_input("Spirit Servings", min_value=0, max_value=400, value=50)
    wine_servings = st.number_input("Wine Servings", min_value=0, max_value=400, value=20)

    continent = st.selectbox(
        "Select Continent",
        ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    )

    predict_button = st.button("Predict Alcohol Consumption")

    if predict_button:
        # One-hot encode continent for model
        continent_data = {
            "continent_Africa": 0, "continent_Asia": 0, "continent_Europe": 0,
            "continent_North America": 0, "continent_Oceania": 0, "continent_South America": 0
        }
        continent_key = f"continent_{continent}"
        if continent_key in continent_data:
            continent_data[continent_key] = 1

        input_data = pd.DataFrame([{
            "beer_servings": beer_servings,
            "spirit_servings": spirit_servings,
            "wine_servings": wine_servings,
            **continent_data
        }])

        prediction = model.predict(input_data)[0]

        st.write("---")
        st.header("Prediction Result")
        st.metric(
            label="Predicted Pure Alcohol (Litres)",
            value=f"{prediction:.2f} L"
        )

        # --- PIE CHART SECTION STARTS HERE ---
        st.subheader("Your Beverage Mix")
        labels = ['Beer', 'Spirits', 'Wine']
        sizes = [beer_servings, spirit_servings, wine_servings]
        colors = ['#F4D03F', '#85C1E9', '#C0392B']

        if sum(sizes) > 0:
            fig3, ax3 = plt.subplots(figsize=(5, 5))
            # textprops={'color':"w"} makes the text white to read better on dark themes
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'color':"w"})
            ax3.axis('equal') 
            fig3.patch.set_alpha(0) # Makes background transparent
            st.pyplot(fig3)
        else:
            st.warning("Please enter some servings to see your mix!")
        # --- PIE CHART SECTION ENDS HERE ---


# RIGHT COLUMN: Charts Only
with output_col:
    st.header("📊 Alcohol Servings by Continent")

    # Horizontal Beer chart
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df, y="continent", x="beer_servings", ax=ax1, color="#F4D03F", errorbar=None)
    ax1.set_xlabel("Beer Servings")
    ax1.set_ylabel("")
    ax1.set_title("Average Beer Servings by Continent", color="#B9770E")
    st.pyplot(fig1)

    # Horizontal Wine chart
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df, y="continent", x="wine_servings", ax=ax2, color="#C0392B", errorbar=None)
    ax2.set_xlabel("Wine Servings")
    ax2.set_ylabel("")
    ax2.set_title("Average Wine Servings by Continent", color="#922B21")
    st.pyplot(fig2)

# --- Small Model Info to fill the space ---
# --- Small Model Info to fill the space ---
    st.write("---")
    with st.expander("ℹ️ Model Information"):
        st.write("This prediction is based on a **Linear Regression** model.")
        st.write("**R² Score:** 0.85") # Use your real score
        st.write("Model selected after comparing with Random Forest.")


    