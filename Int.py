import streamlit as st
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    # load dataset
    dataset = pd.read_csv("Db_sentenze_1.csv")
    return dataset


model = pickle.load(open('model.pkl', 'rb'))

dataset = load_data()
dataset = dataset.drop(dataset.columns[0], axis=1)
random_rows = dataset.sample(n=10, random_state=42)

software_mapping = {
    1: 'Odoo',
    2: 'SAP Business One',
    3: 'Dynamics365',
    4: 'Bitrix24',
    5: 'NetSuite',
    6: 'TeamSystem Enterprise',
    7: 'Fluentis ERP',
    8: 'Sage 100'
}

# Assign the correct software mapping to the selected phrases.
random_rows['Software'] = random_rows['Software'].map(software_mapping)

# Streamlit application
st.title(":green[Streamlit App]")

st.markdown("Seleziona la frase più adatta ai tuoi bisogni:")

# Check if button_clicks.csv file exists
if os.path.isfile("button_clicks.csv"):
    button_clicks = pd.read_csv("button_clicks.csv")
else:

    button_clicks = pd.DataFrame(columns=['Text', 'Software'])

for index, row in random_rows.iterrows():
    button_key = f"button_{index}"
    if st.button(row['Text'], key=button_key):

        st.write("Software corrispondente:", row['Software'])

        button_clicks = button_clicks.append(row[['Text', 'Software']], ignore_index=True)

# Display button click information
if not button_clicks.empty:
    st.markdown(":green[Informazioni:]")
    st.write(button_clicks)

# Save button click information to button_clicks.csv file
button_clicks.to_csv("button_clicks.csv", index=False)

st.markdown(":green[Non hai trovato nessuna frase che ti soddisfa.]")


def predict(payload):
    scores = model.predict_proba([payload])[0]
    return [{"label": software_mapping[i+1], "score": score.item()} for i, score in enumerate(scores)]

def main():
    st.title(":green[Search:]")
    example = st.text_input("Inserisci il tuo bisogno:")
    if st.button("Cerca"):
        results = predict(example)
        risultati = pd.DataFrame(results)

        st.write(risultati)
		
		# Plot the scores
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.barh(risultati['label'], risultati['score'])
        ax.set_xlabel('Score')
        ax.set_ylabel('Software')
        ax.set_title('Prediction Scores')
        
        # Display the plot using st.pyplot()
        st.pyplot(fig)


if __name__ == "__main__":
	st.set_option('deprecation.showPyplotGlobalUse', False)
	main()
