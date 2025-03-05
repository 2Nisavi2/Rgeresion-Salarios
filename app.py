import gradio as gr
import joblib

# Cargar modelo y transformer
model = joblib.load("Model/model_poly.pkl")
transformer = joblib.load("Model/poly_transformer.pkl")

# Función de predicción con manejo de errores
def predict(input_data):
    try:
        transformed_data = transformer.transform([[input_data]])
        prediction = model.predict(transformed_data)
        return str(prediction[0])
    except Exception as e:
        return f"Error: {str(e)}"

# Crear interfaz web
interface = gr.Interface(fn=predict, inputs="text", outputs="text")
interface.launch(share=True)