
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

import pickle
import logging

#Imposto il logger in modo tale che scriva sia su console che su un file. Voglio loggare tutte le richieste quindi utilizzo il livello debug.
logging.basicConfig(level=logging.DEBUG, format=" %(name)s - %(asctime)s - %(message)s", datefmt="%d/%m/%Y %H:%M:%S",
                    handlers=[logging.FileHandler("audit.log", mode="a", encoding="utf-8"), logging.StreamHandler()]
                    )
logger = logging.getLogger(__name__)

#Definisco i modelli dati.

#Il body sarà composto da un attributo text obbligatorio
class PredictBody(BaseModel):
    review: str = Field(..., title="Testo di cui identificare il sentiment", example="Questo è un esempio di testo.")

#L'output sarà un oggetto contenente  il codice della lingua e la probabilità associata
class PredictionOutput(BaseModel):
    sentiment: str = Field(..., title="Sentiment predetto")
    confidence: float = Field(..., title="Probabilità associata alla predizione")


#Carico il modello di Machine Learning pubblicato su github precedentemento scaricato nella mia cartella di lavoro
path_model = 'sentiment_analysis_model.pkl'

with open(path_model, 'rb') as f:
    model = pickle.load(f)


#Inizializzo l'API usando la libreira FastAPI
app = FastAPI(title="SentimentAnalysisAPI",description="API per il Riconoscimento del sentiment di un testo")


#Creo l'endpoind di tipo post che prende in input un oggetto di tipo PredictBody e restituisce un output di tipo PredictionOutput
@app.post("/predict", tags=["SentimentAnalysisAPI"])
def predict_sentiment(body: PredictBody):

    # Controllo se il testo è vuoto o composto solo da spazi
    if body.review is None or body.review.strip() == '':
        logger.error("Ricevuto input con testo vuoto")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST ,detail="Il testo ricevuto è vuoto")

    logger.info(f"Ricevuto il messaggio {body.review}")

    try:
        # Predizione del sentiment
        predict_sentiment = model.predict([body.review])[0]

        # Calcolo della probabilità
        predicted_sentiment_proba = model.predict_proba([body.review])[0]
        confidence = round(float(predicted_sentiment_proba.max()),
                           2)  # Arrotondato a 2 decimali come da esempio della traccia

       
        #Se lìil valore di confidence è minore del 50%, decido anche in questo caso di lanciare un errore,
        # in quantò la probabilità è troppo bassa
        if confidence<0.5:
            logger.error(f"Sentiment identificato: {predict_sentiment} - Confidence: {confidence}. La confidence è troppo bassa!!")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il testo ricevuto non è sufficientemente adeguato per predire la lingua")

        # Creazione dell'output
        response_data = PredictionOutput(sentiment=predict_sentiment, confidence=confidence)

        # Loggo la risposta
        logger.info(f"Sentiment identificato: {predict_sentiment} - Confidence: {confidence}")

        return response_data
    except HTTPException as e:
        # Gestione errore nel caso in cui la lingua non possa essere identificata
        logger.error(f"Errore di identificazione del sentiemnt: {str(e)}")
        raise e
    except Exception as e:
        # Gestione errore generico
        logger.error(f"Errore generico: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Si è verificato un errore imprevisto"
        )




if __name__ == "__main__":
    uvicorn.run("sentiment_analysis:app", reload=True)