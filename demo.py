import streamlit as st
import joblib
import pandas as pd
from os.path import dirname, join, realpath


st.header("Laduma Analytics Football League Winners")

st.subheader("""A simple app to predict results of a football match""")


my_form = st.form(key="Match Form")
season = my_form.selectbox('Season',('1','2'))
hometeam = my_form.selectbox(
     'Home Team',
     ('Antennae', 'Andromeda', 'Butterfly','Cartwheel','Sculptor','Cigar','Comet','Cosmos Redshift 7','Eye of Sauron','Medusa Merger','Milky Way','Sunflower','Tadpole','Fireworks','Backward','Circinus','Coma Pinwheel','Sombrero','Triangulum'))

awayteam = my_form.selectbox(
     'Away Team',
     ('Antennae', 'Andromeda', 'Butterfly','Cartwheel','Sculptor','Cigar','Comet','Cosmos Redshift 7','Eye of Sauron','Medusa Merger','Milky Way','Sunflower','Tadpole','Fireworks','Backward','Circinus','Coma Pinwheel','Sombrero','Triangulum'))

submit = my_form.form_submit_button(label="Predict Results")

# load the model and one-hot-encoder and scaler

with open(
    join(dirname(realpath(__file__)), "lgbmc_model.pkl"),
    "rb",
) as f:
    model = joblib.load(f)

with open(
    join(dirname(realpath(__file__)), "le_encoding.pkl"), "rb"
) as f:
    leencoding= joblib.load(f)


with open(
    join(dirname(realpath(__file__)), "minmaxscaler.pkl"), "rb"
) as f:
    scaler = joblib.load(f)

@st.cache
# function to clean and tranform the input
def preprocessing_data(data, enc, scaler):

    # Convert the following numerical labels from integer to float
#float_array= data[["Season_y"]].values.astype(float)

    # One Hot Encoding conversion
 data = enc.transform(data)

    # scale our data into range of 0 and 1
 data = scaler.transform(data)

 return data


if submit:

    # collect inputs
    input = {
        "Season": season,
        "Home Team": hometeam,
        "Away Team": awayteam
    }

    # create a draframe
    data = pd.DataFrame(input, index=[0])

    # clean and transform input
    transformed_data = preprocessing_data(data=data, enc=leencoding, scaler=scaler)

    # perform prediction
    prediction = model.predict(transformed_data)
    output = int(prediction[0])
    probas = model.predict_proba(transformed_data)
    probability = "{:.2f}".format(float(probas[:, output]))

    # Display results of the NLP task
    st.header("Results")
    if output == 0:
        st.write(
            "The result is Home Team Wins with probability of {}".format(probability)
        )
    elif output == 1:
        st.write(
             "The result is Draw with probability of {}".format(probability)
            )
    elif output == 2:
        st.write(
             "The result is Away Team Wins with probability of {}".format(probability)
        )

