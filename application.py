import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def predictSalary(experiance):
    try:
        # Reading data(csv file)
        data = pd.read_csv('Data/Salary_dataset.csv', index_col=0)
        # Checking any null values
        if data.iloc[:,0].isnull().any() == True:
            if data.shape[0] < 200:
                data = data.fillna(0)
            else:
                data = data.dropna()

        # # Splitting the dependent and predicting columns
        x = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values.reshape(-1, 1)

        # Scalling the dataset for good prediction
        scaler = StandardScaler()
        x_sc = scaler.fit_transform(x)

        # Training the model
        model = LinearRegression()
        model.fit(x_sc, y)

        exp = [[float(experiance)]]
        
        prediction = model.predict(scaler.transform(exp))
        conv_pred = int(prediction)
        roundOf_pred = round(conv_pred, 2)
        return roundOf_pred
    except Exception as error:
        print(error)


iface = gr.Interface(
    title="Salary Prediction",
    fn = predictSalary,
    inputs = [
        gr.Number(value=0,label='Total Experiance')
    ],
    outputs = [
        gr.Number(label='Predicted Salary')
    ]
)

if __name__ == "__main__":
    iface.launch()